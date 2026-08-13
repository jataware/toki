import time
import warnings
from typing import Literal

from ..helpers.cache_state import _CacheState, estimate_messages_tokens
from ..litellm.model import _LiteLLMModel
from ..model import ReasoningEffort, TokenCountEstimate, TokiCacheWarning, TokiMessage, ToolsArg
from .models import GoogleModelName, attributes_map


_GOOGLE_OFFLINE_SAFETY_FACTOR_DEFAULT = 1.15


# Default knobs for explicit-cache behavior. Gemini 3 Pro requires a 4096-token
# floor for explicit caching; the older 2.5 family is at 1024. We pick the
# upper figure so the same defaults work across the model lineup.
_GOOGLE_CACHE_DEFAULT_TTL = 3600
_GOOGLE_CACHE_DEFAULT_MIN_TOKENS = 4096
_GOOGLE_CACHE_DEFAULT_REFRESH_DELTA = 4096
_GOOGLE_CACHE_DEFAULT_REFRESH_BUFFER = 60


def _toki_messages_to_genai(messages: list[TokiMessage]) -> tuple[str | None, list[dict]]:
    """Translate a slice of toki messages into Gemini `Content` dicts.

    System messages are concatenated and returned as a `system_instruction`
    string (Gemini's caches separate it from the conversation). Assistant
    becomes role=`'model'`; tool calls become `function_call` parts; tool
    results become `function_response` parts under a synthesized `'user'` role
    (matches Gemini's tool-calling conventions).
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        if m.role == 'system':
            if m.content:
                system_parts.append(m.content)
            continue
        if m.role == 'user':
            contents.append({"role": "user", "parts": [{"text": m.content or ""}]})
            continue
        if m.role == 'assistant':
            parts: list[dict] = []
            if m.content:
                parts.append({"text": m.content})
            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append({"function_call": {"name": tc.function.name, "args": tc.function.arguments}})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            continue
        if m.role == 'tool':
            contents.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": m.tool_call_id or "",
                        "response": {"result": m.content or ""},
                    }
                }],
            })
            continue
    system = "\n".join(system_parts) if system_parts else None
    return system, contents


def _wire_tools_to_genai(wire_tools: list[dict] | None) -> list[dict] | None:
    """Translate OpenAI-style tool dicts into Gemini's
    `tools=[{'function_declarations': [...]}]` shape."""
    if not wire_tools:
        return None
    function_declarations: list[dict] = []
    for t in wire_tools:
        fn = t.get("function") or {}
        decl: dict = {"name": fn.get("name")}
        if fn.get("description") is not None:
            decl["description"] = fn["description"]
        if fn.get("parameters") is not None:
            decl["parameters"] = fn["parameters"]
        function_declarations.append(decl)
    return [{"function_declarations": function_declarations}]


class _GoogleCacheManager:
    """Drives the explicit-cache lifecycle for `GoogleModel`.

    Wraps a `_CacheState` (anchor history, hash-based prefix lookup) plus a
    lazy `google.genai.Client`. Per call, it asks `_CacheState` for an entry,
    creates a `cachedContents/<id>` resource via the Google SDK if the entry
    has no live cache yet, and returns the (tail_messages, cache_name) pair
    the model should pass through to litellm.

    Cache creation is best-effort: any SDK exception (`'too small'`, model not
    supported, quota, network) is caught, warned about, and falls back to a
    non-cached call for that turn.
    """

    def __init__(
        self,
        *,
        sdk_model_name: str,
        api_key: str,
        ttl_seconds: int,
        min_tokens: int,
        refresh_delta_tokens: int,
        refresh_buffer_seconds: int,
    ):
        self._sdk_model = sdk_model_name
        self._api_key = api_key
        self._ttl = ttl_seconds
        self._refresh_delta = refresh_delta_tokens
        self._refresh_buffer = refresh_buffer_seconds
        self.state = _CacheState(min_cache_size_estimate=min_tokens)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for explicit caching on GoogleModel. "
                "Install with `pip install toki[google]`."
            ) from e
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _is_alive(self, entry) -> bool:
        if entry.cache_name is None or entry.expires_at is None:
            return False
        return entry.expires_at - time.time() > self._refresh_buffer

    def _build_create_kwargs(self, prefix_messages: list[TokiMessage], tools_wire: list[dict] | None) -> dict | None:
        """Returns the `client.caches.create` kwargs, or `None` if the prefix
        translates to empty `contents` (Gemini rejects caches with no
        non-system content — happens in rolling mode when the only thing to
        cache is the system prompt)."""
        system, contents = _toki_messages_to_genai(prefix_messages)
        if not contents:
            return None
        config: dict = {"contents": contents, "ttl": f"{self._ttl}s"}
        if system:
            config["system_instruction"] = system
        genai_tools = _wire_tools_to_genai(tools_wire)
        if genai_tools:
            config["tools"] = genai_tools
        return {"model": self._sdk_model, "config": config}

    @staticmethod
    def _expires_from_cache(cache) -> float:
        et = getattr(cache, "expire_time", None)
        if et is None:
            return time.time() + 3600
        if hasattr(et, "timestamp"):
            return et.timestamp()
        return float(et)

    def _resolve_entry(self, *, strategy, messages, tools_wire) -> tuple[object | None, list[TokiMessage] | None]:
        candidate_anchor = len(messages) if strategy == 'static' else len(messages) - 1
        if candidate_anchor <= 0:
            return None, None
        prefix_estimate = estimate_messages_tokens(None, tools_wire, messages[:candidate_anchor])
        entry = self.state.match_or_snapshot(
            strategy=strategy,
            messages=messages,
            system=None,
            tools=tools_wire,
            prefix_token_estimate=prefix_estimate,
            refresh_delta_tokens=self._refresh_delta if strategy == 'rolling' else 0,
        )
        if entry is None:
            return None, None
        return entry, messages[: entry.anchor_index]

    def prepare(
        self,
        *,
        strategy: Literal['rolling', 'static'],
        messages: list[TokiMessage],
        tools_wire: list[dict] | None,
    ) -> tuple[list[TokiMessage] | None, str | None]:
        """Sync path. Returns (tail_messages, cache_name); both `None` means
        no caching this turn."""
        entry, prefix = self._resolve_entry(strategy=strategy, messages=messages, tools_wire=tools_wire)
        if entry is None:
            return None, None
        if not self._is_alive(entry):
            create_kwargs = self._build_create_kwargs(prefix, tools_wire)
            if create_kwargs is None:
                return None, None
            try:
                client = self._ensure_client()
                cache = client.caches.create(**create_kwargs)
            except Exception as e:
                warnings.warn(
                    f"Gemini explicit-cache creation failed ({type(e).__name__}: {e}); "
                    "falling back to a non-cached call.",
                    category=TokiCacheWarning,
                    stacklevel=4,
                )
                return None, None
            entry.cache_name = cache.name
            entry.expires_at = self._expires_from_cache(cache)
        return messages[entry.anchor_index:], entry.cache_name

    async def aprepare(
        self,
        *,
        strategy: Literal['rolling', 'static'],
        messages: list[TokiMessage],
        tools_wire: list[dict] | None,
    ) -> tuple[list[TokiMessage] | None, str | None]:
        """Async path: uses `client.aio.caches.create` for cache creation."""
        entry, prefix = self._resolve_entry(strategy=strategy, messages=messages, tools_wire=tools_wire)
        if entry is None:
            return None, None
        if not self._is_alive(entry):
            create_kwargs = self._build_create_kwargs(prefix, tools_wire)
            if create_kwargs is None:
                return None, None
            try:
                client = self._ensure_client()
                cache = await client.aio.caches.create(**create_kwargs)
            except Exception as e:
                warnings.warn(
                    f"Gemini explicit-cache creation failed ({type(e).__name__}: {e}); "
                    "falling back to a non-cached call.",
                    category=TokiCacheWarning,
                    stacklevel=4,
                )
                return None, None
            entry.cache_name = cache.name
            entry.expires_at = self._expires_from_cache(cache)
        return messages[entry.anchor_index:], entry.cache_name


class GoogleModel(_LiteLLMModel):
    """Toki frontend for Google's Gemini models via AI Studio, dispatched through litellm.

    Uses litellm's `gemini/*` provider with `GEMINI_API_KEY`. Vertex AI is a
    separate provider in litellm and is not wired up here.

    Model ids match AI Studio / Gemini API names (e.g. `gemini-2.5-flash`).
    The live catalog is https://ai.google.dev/gemini-api/docs/models. The
    snapshot bundled with this install is `GoogleModelName` /
    `toki.google.list_google_models`; ids outside that Literal still work if
    AI Studio serves them.

    Caching is opt-in via `cache=`:

      - `cache='rolling'` — every turn (re)creates a `cachedContents` resource
        covering `messages[:-1]` once the prefix is large enough, sending only
        the latest message live. The cache is reused across calls until the
        conversation grows by `cache_refresh_delta_tokens` (chars/4 estimate)
        or the cache nears expiry.
      - `cache='static'` — pins a `cachedContents` resource at the first call
        whose prefix estimate clears `cache_min_tokens`, then reuses it for
        every subsequent call (recreating only when its TTL is about to
        expire). All messages past the anchor are sent live.
      - `cache=None` (default) — no explicit-cache work; Gemini's *implicit*
        caching (automatic on 2.5+/3.x models) still applies.

    `model.cache` is freely mutable mid-session; `model.invalidate_cache()`
    forgets the historical anchor list.
    """

    def __init__(
        self,
        model: GoogleModelName | str,
        *,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
        cache: Literal['rolling', 'static'] | None = None,
        cache_ttl: int = _GOOGLE_CACHE_DEFAULT_TTL,
        cache_min_tokens: int = _GOOGLE_CACHE_DEFAULT_MIN_TOKENS,
        cache_refresh_delta_tokens: int = _GOOGLE_CACHE_DEFAULT_REFRESH_DELTA,
        cache_refresh_buffer_seconds: int = _GOOGLE_CACHE_DEFAULT_REFRESH_BUFFER,
    ):
        super().__init__(
            wire_model=f"gemini/{model}",
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )
        self.model = model
        self.cache = cache
        self._cache_manager = _GoogleCacheManager(
            sdk_model_name=f"models/{model}",
            api_key=api_key,
            ttl_seconds=cache_ttl,
            min_tokens=cache_min_tokens,
            refresh_delta_tokens=cache_refresh_delta_tokens,
            refresh_buffer_seconds=cache_refresh_buffer_seconds,
        )

    def _attributes_map(self) -> dict:
        return attributes_map

    def invalidate_cache(self) -> None:
        """Drop the historical anchor list. Next `'static'` call snapshots
        from scratch; rolling immediately recreates a fresh cache."""
        self._cache_manager.state.clear()

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _GOOGLE_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        """Count the prompt tokens for the given messages (and tools).
        
        Two modes:

          - `kind='exact'` / `kind='online'` — issues a
            `max_tokens=1` chat completion via litellm and reads
            `usage.prompt_tokens` off the response. Returns a plain `int`
            equal to the number Gemini would charge. Costs the prompt +
            one output token per call. (Gemini ships a dedicated
            `count_tokens` endpoint, but on AI Studio it accepts only
            `contents` — no `tools`, no `system_instruction` — so toki
            uniformly routes through a generation call instead.)
          - `kind='offline'` — runs `litellm.token_counter` locally and
            wraps the heuristic count in a `TokenCountEstimate`
            (`prompt_tokens`, `raw_prompt_tokens`, `safety_factor`). Google
            doesn't ship an official offline tokenizer; `safety_factor`
            (default 1.15) multiplies the raw count to give a budget-safe
            figure.

        Any other `kind` value raises `ValueError`. `safety_factor` only
        applies on the offline path.
        """
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"GoogleModel.count_tokens: unsupported kind {kind!r}")
        wire_messages, wire_tools = self._normalize_for_count(messages, tools)
        if kind == 'offline':
            raw = self._litellm_offline_count(wire_messages, wire_tools)
            return self._wrap_estimate(raw, safety_factor)
        return self._litellm_online_count(wire_messages, wire_tools)

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _GOOGLE_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        """Async sibling of `count_tokens`. Same behavior; the online path
        uses `litellm.acompletion` so it doesn't block the event loop. The
        offline path is pure-CPU work and runs inline."""
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"GoogleModel.acount_tokens: unsupported kind {kind!r}")
        wire_messages, wire_tools = self._normalize_for_count(messages, tools)
        if kind == 'offline':
            raw = self._litellm_offline_count(wire_messages, wire_tools)
            return self._wrap_estimate(raw, safety_factor)
        return await self._litellm_online_count_async(wire_messages, wire_tools)

    def _post_cache_kwargs(self, tail: list[TokiMessage], cache_name: str, kwargs: dict, *, capture_thinking: bool) -> tuple[list[dict], dict]:
        """Build the live request shape after a successful cache resolution:
        strip system messages from the tail (they're inside the cache),
        attach `cached_content`, and drop `tools` / `parallel_tool_calls`
        (they're inside the cache too)."""
        from ..litellm.model import _msg_to_wire
        wire_messages = [_msg_to_wire(m) for m in tail if m.role != 'system']
        out_kwargs = dict(kwargs)
        out_kwargs["cached_content"] = cache_name
        out_kwargs.pop("tools", None)
        out_kwargs.pop("parallel_tool_calls", None)
        if self.reasoning_effort is not None and "reasoning_effort" not in out_kwargs and "thinking" not in out_kwargs:
            out_kwargs["reasoning_effort"] = self.reasoning_effort
        elif capture_thinking and "reasoning_effort" not in out_kwargs and "thinking" not in out_kwargs:
            out_kwargs["reasoning_effort"] = "medium"
        return wire_messages, out_kwargs

    def _prepare_call(self, messages, tools, kwargs, *, capture_thinking: bool = False):
        if self.cache is None:
            return super()._prepare_call(messages, tools, kwargs, capture_thinking=capture_thinking)
        tail, cache_name = self._cache_manager.prepare(
            strategy=self.cache,
            messages=messages,
            tools_wire=tools,
        )
        if cache_name is None:
            return super()._prepare_call(messages, tools, kwargs, capture_thinking=capture_thinking)
        return self._post_cache_kwargs(tail, cache_name, kwargs, capture_thinking=capture_thinking)

    async def _aprepare_call(self, messages, tools, kwargs, *, capture_thinking: bool = False):
        if self.cache is None:
            return super()._prepare_call(messages, tools, kwargs, capture_thinking=capture_thinking)
        tail, cache_name = await self._cache_manager.aprepare(
            strategy=self.cache,
            messages=messages,
            tools_wire=tools,
        )
        if cache_name is None:
            return super()._prepare_call(messages, tools, kwargs, capture_thinking=capture_thinking)
        return self._post_cache_kwargs(tail, cache_name, kwargs, capture_thinking=capture_thinking)
