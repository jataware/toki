import json
import os
import warnings
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, Literal, TypedDict, cast

import httpx
import requests
import urllib3
from requests.adapters import HTTPAdapter
from typing_extensions import NotRequired

from ..anthropic.utils import _with_text_cache_marker, apply_cache_markers, build_cache_control
from ..helpers.cache_state import _CacheState, estimate_messages_tokens
from ..model import (
    BaseModel,
    ReasoningEffort,
    TokenCountEstimate,
    TokiCacheWarning,
    TokiMessage,
    TokiThinkingSupportWarning,
    TokiToolCall,
    TokiUsageMetadata,
    ToolsArg,
    _RawChunk,
    _RawContentChunk,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
    _unwrap_tools,
)
from .models import OpenRouterModelName, attributes_map


_OPENROUTER_OFFLINE_SAFETY_FACTOR_DEFAULT = 1.15


# OpenRouter (like Anthropic) requires the cacheable prefix to clear ~1024
# tokens before any cache write happens. We use the same chars/4 estimate
# the rest of toki does.
_OPENROUTER_MIN_CACHE_TOKENS = 1024


# Network timeouts, in seconds (`None` disables that limit). Overridable per
# instance via the matching `OpenRouterModel` constructor kwargs.
#
#   - connect: TCP + TLS setup, applied per resolved address. urllib3 walks
#     getaddrinfo's results in order, so an address that black-holes (e.g. an
#     unroutable IPv6 AAAA record) costs one connect timeout and then the next
#     address is tried. Without a limit the first address hangs forever.
#   - read: longest silence tolerated from the server. For non-streaming calls
#     that is roughly time-to-full-response, which on reasoning models can
#     legitimately be several minutes, hence the generous default.
#   - stream read: longest gap between SSE chunks on streaming calls. OpenRouter
#     sends keep-alive comments while the upstream is still working, so this can
#     be much tighter than the non-streaming read timeout.
_OPENROUTER_CONNECT_TIMEOUT_DEFAULT = 10.0
_OPENROUTER_READ_TIMEOUT_DEFAULT = 600.0
_OPENROUTER_STREAM_READ_TIMEOUT_DEFAULT = 60.0

# Connections kept alive per `OpenRouterModel` (sync path). Above urllib3's
# default of 10 so threaded callers sharing one model still reuse connections
# instead of dialing (and paying any connect-timeout fallback) again.
_OPENROUTER_POOL_MAXSIZE = 32


class OpenRouterReasoningDetail(TypedDict):
    type: str
    text: NotRequired[str]


class OpenRouterMessagePayload(TypedDict):
    role: str
    content: NotRequired[str]
    tool_calls: NotRequired[list[dict]]
    reasoning: NotRequired[str]
    reasoning_details: NotRequired[list[OpenRouterReasoningDetail]]


class OpenRouterResponseCompletionChoice(TypedDict):
    message: OpenRouterMessagePayload


class OpenRouterResponse(TypedDict):
    choices: list[OpenRouterResponseCompletionChoice]
    usage: dict  # OpenAI-shaped usage object; see `_usage_from_wire`


class OpenRouterResponseDeltaPayload(TypedDict):
    content: NotRequired[str]
    tool_calls: NotRequired[list[dict]]
    reasoning: NotRequired[str]
    reasoning_details: NotRequired[list[OpenRouterReasoningDetail]]


class OpenRouterResponseChoice(TypedDict):
    delta: OpenRouterResponseDeltaPayload


class OpenRouterResponseDelta(TypedDict):
    choices: list[OpenRouterResponseChoice]
    usage: NotRequired[dict]  # typically only on the final chunk; see `_usage_from_wire`


class OpenRouterResponseError(TypedDict):
    error: Any


_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _tool_call_to_wire(tc: TokiToolCall) -> dict:
    # OpenRouter / OpenAI wire format wants `arguments` as a JSON-encoded string
    return {
        "id": tc.id,
        "type": tc.type,
        "function": {"name": tc.function.name, "arguments": json.dumps(tc.function.arguments)},
    }


def _msg_to_wire(m: TokiMessage) -> dict:
    """Serialize a TokiMessage dataclass to the OpenRouter wire shape, dropping None fields."""
    out: dict = {"role": m.role, "content": m.content}
    if m.tool_calls is not None:
        out["tool_calls"] = [_tool_call_to_wire(tc) for tc in m.tool_calls]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _extract_reasoning_text(payload: OpenRouterMessagePayload | OpenRouterResponseDeltaPayload) -> str:
    """Pull reasoning text out of a delta or message payload. Prefers reasoning_details (skipping
    encrypted/non-text entries); falls back to the plain `reasoning` string field. Returns "" if absent."""
    details = payload.get("reasoning_details")
    if details:
        parts = [d.get("text", "") for d in details if d.get("type") == "reasoning.text"]
        text = "".join(parts)
        if text:
            return text
    return payload.get("reasoning", "") or ""


class OpenRouterModel(BaseModel):
    """Toki model backend that talks to OpenRouter's chat-completions API over HTTPS.

    Model ids are `provider/model` slugs (e.g. `anthropic/claude-sonnet-4`).
    The live catalog is https://openrouter.ai/models. The snapshot bundled
    with this install (IDE autocomplete) is `OpenRouterModelName` /
    `toki.openrouter.list_openrouter_models`; ids outside that Literal still
    work if OpenRouter serves them.

    Caching is opt-in via `cache=`:

      - `anthropic/*` route — `'rolling'` adds a top-level `cache_control`
        breakpoint that OpenRouter auto-advances each turn; `'static'` places
        explicit per-block markers via the shared `apply_cache_markers`
        helper at a snapshotted anchor point.
      - `google/*` route — places a single `cache_control` marker on the
        latest user message (`'rolling'`) or on the snapshot anchor message
        (`'static'`). OpenRouter manages cache lifecycle server-side; the
        `cache_ttl` kwarg has no effect on this route (Gemini caches default
        to ~5 minutes).
      - Any other prefix — `cache=` triggers a `UserWarning` at construction
        because the upstream provider doesn't honor caching breakpoints.

    `model.cache` is a regular mutable attribute and `model.invalidate_cache()`
    drops the historical anchor list. The `_CacheState` helper retains every
    historical anchor so reverting to a previously-cached prefix silently
    rehydrates without re-snapshotting.

    Network timeouts (seconds; `None` disables that limit), all mutable
    attributes like `cache`:

      - `connect_timeout` (default 10) — per resolved address, so a
        black-holed address (e.g. broken IPv6) costs this much before the
        next address is tried.
      - `read_timeout` (default 600) — longest server silence on non-streaming
        calls (completions and online `count_tokens`); effectively the
        time-to-response budget, so keep it generous for reasoning models.
      - `stream_read_timeout` (default 60) — longest gap between chunks on
        streaming calls.

    A timeout raises the transport's own exception type (`requests.ConnectTimeout`
    / `requests.ReadTimeout` on the sync path, `httpx.ConnectTimeout` /
    `httpx.ReadTimeout` etc. on the async path) with a message naming the
    limit that fired. Sync calls reuse pooled keep-alive connections from a
    per-instance `requests.Session`, so only a new connection pays a
    connect-timeout fallback.
    """

    def __init__(
        self,
        model: OpenRouterModelName,
        api_key: str,
        allow_parallel_tool_calls: bool = False,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        cache: Literal['rolling', 'static'] | None = None,
        cache_ttl: Literal['5m', '1h'] = '5m',
        connect_timeout: float | None = _OPENROUTER_CONNECT_TIMEOUT_DEFAULT,
        read_timeout: float | None = _OPENROUTER_READ_TIMEOUT_DEFAULT,
        stream_read_timeout: float | None = _OPENROUTER_STREAM_READ_TIMEOUT_DEFAULT,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.reasoning_effort = reasoning_effort
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.stream_read_timeout = stream_read_timeout
        # created lazily by `_session()`; tagged with the creating pid so a
        # forked child never shares the parent's pooled sockets
        self._requests_session: requests.Session | None = None
        self._requests_session_pid: int | None = None
        self._cache_state = _CacheState(min_cache_size_estimate=_OPENROUTER_MIN_CACHE_TOKENS)
        if cache is not None and self._cache_route() is None:
            warnings.warn(
                f"OpenRouter caching has no effect for {model!r}; provider does not support caching breakpoints.",
                category=TokiCacheWarning,
                stacklevel=2,
            )
        # rolling cache on Anthropic-route models engages caching every turn
        # but does not reliably produce cache reads on Claude. Surface this once
        # at construction so the user knows to switch to 'static' if they need
        # deterministic prefix-cache hits.
        if cache == 'rolling' and model.startswith('anthropic/'):
            warnings.warn(
                f"OpenRouterModel cache='rolling' on Anthropic-route model {model!r}: "
                "rolling caching engages cache_control breakpoints every turn but does "
                "not reliably produce cache reads on Claude through OpenRouter. Use "
                "cache='static' if you need deterministic prefix-cache hits.",
                category=TokiCacheWarning,
                stacklevel=2,
            )

    def _get_allow_parallel_tool_calls(self) -> bool:
        return self.allow_parallel_tool_calls

    def _supports_thinking(self) -> bool | None:
        attr = attributes_map.get(self.model)
        if attr is None:
            return None
        return getattr(attr, 'supports_thinking', None)

    def _maybe_warn_capture_thinking(self) -> None:
        # OpenRouter-routed OpenAI models also don't reliably surface reasoning
        # text on the chat-completions wire — same caveat as the native OpenAI
        # backend; fire the same warning before delegating to the base check.
        if self.model.startswith('openai/'):
            self._maybe_warn(
                'capture_thinking_openai_unreliable',
                f"capture_thinking=True on OpenAI-route model {self.model!r}: OpenAI's "
                "chat-completions API does not reliably surface reasoning text — "
                "server-side reasoning still engages (and improves answer quality at "
                "higher effort), but the chain itself is rarely returned. Silence via "
                "`warnings.filterwarnings('ignore', category=toki.TokiThinkingSupportWarning)`.",
                category=TokiThinkingSupportWarning,
                stacklevel=4,
            )
        super()._maybe_warn_capture_thinking()

    def invalidate_cache(self) -> None:
        """Drop the historical anchor list. Next `'static'` call will defer
        until the prefix is large enough and snapshot afresh."""
        self._cache_state.clear()

    # ----- token counting ---------------------------------------------------

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _OPENROUTER_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        """
        Count the prompt tokens for the given messages (and tools).

        Two modes:

          - `kind='exact'` / `kind='online'` — POSTs a
            `max_tokens=1` `/chat/completions` request to OpenRouter and
            reads `usage.prompt_tokens` off the response. Returns a plain
            `int` matching what the upstream provider would charge for the
            same prompt. Costs the prompt + one output token per call.
            OpenRouter has no dedicated count-tokens endpoint, so this is
            the only path that produces a guaranteed-exact figure.
          - `kind='offline'` — runs `litellm.token_counter` keyed off the
            upstream model id (e.g. `'anthropic/claude-haiku-4-5'`) and
            wraps the heuristic count in a `TokenCountEstimate`
            (`prompt_tokens`, `raw_prompt_tokens`, `safety_factor`). The
            `litellm` import is lazy: if the package isn't installed,
            `ImportError` is raised pointing at `toki[litellm]`. Accuracy
            depends entirely on whether litellm has a tokenizer for the
            chosen upstream model; `safety_factor` (default 1.15) gives a
            budget-safe multiplier on top.

        Any other `kind` value raises `ValueError`. `safety_factor` only
        applies on the offline path.
        """
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"OpenRouterModel.count_tokens: unsupported kind {kind!r}")
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, _streaming = _unwrap_tools(tools)
        wire_messages = [_msg_to_wire(m) for m in normalized]
        if kind == 'offline':
            raw = _openrouter_offline_count(self.model, wire_messages, wire_tools)
            return TokenCountEstimate(
                prompt_tokens=round(raw * safety_factor),
                raw_prompt_tokens=raw,
                safety_factor=safety_factor,
            )
        payload = _build_count_payload(self.model, wire_messages, wire_tools)
        with self._clarify_timeouts(stream=False), httpx.Client(timeout=self._httpx_timeout(stream=False)) as client:
            r = client.post(_API_URL, headers=self._headers(stream=False), json=payload)
            r.raise_for_status()
            data = r.json()
        return _prompt_tokens_from_count_response(data)

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _OPENROUTER_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        """Async sibling of `count_tokens`. Same behavior; the online path
        uses `httpx.AsyncClient` so it doesn't block the event loop. The
        offline path is pure-CPU work and runs inline."""
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"OpenRouterModel.acount_tokens: unsupported kind {kind!r}")
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, _streaming = _unwrap_tools(tools)
        wire_messages = [_msg_to_wire(m) for m in normalized]
        if kind == 'offline':
            raw = _openrouter_offline_count(self.model, wire_messages, wire_tools)
            return TokenCountEstimate(
                prompt_tokens=round(raw * safety_factor),
                raw_prompt_tokens=raw,
                safety_factor=safety_factor,
            )
        payload = _build_count_payload(self.model, wire_messages, wire_tools)
        with self._clarify_timeouts(stream=False):
            async with httpx.AsyncClient(timeout=self._httpx_timeout(stream=False)) as client:
                r = await client.post(_API_URL, headers=self._headers(stream=False), json=payload)
                r.raise_for_status()
                data = r.json()
        return _prompt_tokens_from_count_response(data)

    def _cache_route(self) -> Literal['anthropic', 'google'] | None:
        if self.model.startswith('anthropic/'):
            return 'anthropic'
        if self.model.startswith('google/'):
            return 'google'
        return None

    def _apply_caching(
        self,
        messages: list[TokiMessage],
        wire_messages: list[dict],
        wire_tools: list[dict] | None,
        tools: list[dict] | None,
    ) -> tuple[list[dict], list[dict] | None, dict]:
        """Returns (wire_messages, wire_tools, payload_extra). `payload_extra`
        carries any top-level fields (only used for the anthropic auto-mode
        cache_control)."""
        route = self._cache_route()
        if self.cache is None or route is None:
            return wire_messages, wire_tools, {}

        if route == 'anthropic':
            if self.cache == 'rolling':
                # OpenRouter auto-advances the breakpoint based on conversation length.
                return wire_messages, wire_tools, {"cache_control": build_cache_control(self.cache_ttl)}
            entry = self._cache_state.match_or_snapshot(
                strategy='static',
                messages=messages,
                system=None,
                tools=tools,
                prefix_token_estimate=estimate_messages_tokens(None, tools, messages),
            )
            if entry is None:
                return wire_messages, wire_tools, {}
            new_msgs, new_tools = apply_cache_markers(
                wire_messages, wire_tools, mode='static',
                anchor_index=entry.anchor_index, ttl=self.cache_ttl,
            )
            return new_msgs, new_tools, {}

        # google route
        marker = build_cache_control('5m')
        if self.cache == 'rolling':
            idx = len(wire_messages) - 1
        else:
            entry = self._cache_state.match_or_snapshot(
                strategy='static',
                messages=messages,
                system=None,
                tools=tools,
                prefix_token_estimate=estimate_messages_tokens(None, tools, messages),
            )
            if entry is None:
                return wire_messages, wire_tools, {}
            idx = entry.anchor_index - 1
        if not (0 <= idx < len(wire_messages)):
            return wire_messages, wire_tools, {}
        new_msgs = list(wire_messages)
        new_msgs[idx] = _with_text_cache_marker(new_msgs[idx], marker)
        return new_msgs, wire_tools, {}

    def _build_payload(self, messages: list[TokiMessage], tools: list[dict] | None, capture_thinking: bool, stream: bool, kwargs: dict) -> dict:
        wire_messages = [_msg_to_wire(m) for m in messages]
        wire_messages, wire_tools, payload_extra = self._apply_caching(messages, wire_messages, tools, tools)
        tool_payload = {"tools": wire_tools, "parallel_tool_calls": self.allow_parallel_tool_calls} if wire_tools else {}
        # Reasoning payload precedence:
        #   1. user `reasoning=...` via **kwargs (handled by `**kwargs` below)
        #   2. ctor `reasoning_effort`
        #   3. `capture_thinking=True` (auto-engages medium effort via `enabled: true`)
        if "reasoning" in kwargs:
            reasoning_payload: dict = {}
        elif self.reasoning_effort is not None:
            reasoning_payload = {"reasoning": {"effort": self.reasoning_effort}}
        elif capture_thinking:
            reasoning_payload = {"reasoning": {"enabled": True}}
        else:
            reasoning_payload = {}
        payload: dict = {
            "model": self.model,
            "messages": wire_messages,
            **tool_payload,
            **reasoning_payload,
            **payload_extra,
            **kwargs,
        }
        if stream:
            payload["stream"] = True
        return payload

    def _headers(self, *, stream: bool) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if stream:
            h["Accept"] = "text/event-stream"
        return h

    # ----- transport ---------------------------------------------------------

    def _session(self) -> requests.Session:
        """Per-instance `requests.Session` so sync calls reuse keep-alive
        connections: only a freshly dialed connection pays connect-timeout
        fallbacks (e.g. a black-holed IPv6 address before IPv4 is tried).
        Created lazily and re-created after `os.fork()`. Every request passes
        its own headers and timeout and toki never reconfigures the session,
        so the only shared mutable state is urllib3's connection pool and
        requests' cookie jar, both lock-protected; concurrent calls from
        several threads on one model are fine."""
        pid = os.getpid()
        if self._requests_session is None or self._requests_session_pid != pid:
            session = requests.Session()
            session.mount("https://", HTTPAdapter(pool_maxsize=_OPENROUTER_POOL_MAXSIZE))
            self._requests_session = session
            self._requests_session_pid = pid
        return self._requests_session

    def _requests_timeout(self, *, stream: bool) -> tuple[float | None, float | None]:
        return (self.connect_timeout, self.stream_read_timeout if stream else self.read_timeout)

    def _httpx_timeout(self, *, stream: bool) -> httpx.Timeout:
        # matches the previous shape: read/write/pool share one budget
        return httpx.Timeout(self.stream_read_timeout if stream else self.read_timeout, connect=self.connect_timeout)

    @contextmanager
    def _clarify_timeouts(self, *, stream: bool) -> Iterator[None]:
        """Re-raise transport timeouts as the same exception type with a
        message naming the OpenRouter limit that fired and how to change it,
        so `except requests.Timeout` / `except httpx.TimeoutException`
        handlers keep working.

        A read stall after the response headers arrived (e.g. mid-SSE-stream)
        surfaces from `requests` as `requests.ConnectionError` wrapping
        urllib3's `ReadTimeoutError`; it keeps that type but gets the same
        read-timeout message."""
        read_attr = 'stream_read_timeout' if stream else 'read_timeout'

        def read_message(e: Exception) -> str:
            return (
                f"OpenRouter request for {self.model!r} got no data within "
                f"{read_attr}={getattr(self, read_attr)}s. Long reasoning calls may need more; "
                f"adjust via OpenRouterModel({read_attr}=...). Cause: {e}"
            )

        try:
            yield
        except (requests.exceptions.ConnectTimeout, httpx.ConnectTimeout) as e:
            raise _with_message(e, (
                f"OpenRouter request for {self.model!r} could not connect to openrouter.ai within "
                f"connect_timeout={self.connect_timeout}s (per resolved address). Check DNS/IPv6 "
                f"routing, or adjust via OpenRouterModel(connect_timeout=...). Cause: {e}"
            )) from e
        except (requests.exceptions.ReadTimeout, httpx.ReadTimeout) as e:
            raise _with_message(e, read_message(e)) from e
        except requests.exceptions.ConnectionError as e:
            if not (e.args and isinstance(e.args[0], urllib3.exceptions.ReadTimeoutError)):
                raise
            raise _with_message(e, read_message(e)) from e
        except httpx.TimeoutException as e:  # write / pool timeouts on the async path
            raise _with_message(e, (
                f"OpenRouter request for {self.model!r} timed out ({type(e).__name__}). "
                f"Adjust via OpenRouterModel({read_attr}=...). Cause: {e}"
            )) from e

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        payload = self._build_payload(messages, tools, capture_thinking, stream=False, kwargs=kwargs)
        with self._clarify_timeouts(stream=False):
            response = self._session().post(
                _API_URL, headers=self._headers(stream=False), json=payload, timeout=self._requests_timeout(stream=False)
            )
        return _turn_from_blocking_json(response.json(), capture_thinking=capture_thinking)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        payload = self._build_payload(messages, tools, capture_thinking, stream=True, kwargs=kwargs)

        with self._clarify_timeouts(stream=True), self._session().post(
            _API_URL, headers=self._headers(stream=True), json=payload, stream=True, timeout=self._requests_timeout(stream=True)
        ) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            buf: list[str] = []
            for line in r.iter_lines(decode_unicode=True, chunk_size=1024):
                line = cast(str | None, line)
                if line is None:
                    continue

                if line.startswith("data:"):
                    buf.append(line[5:].lstrip())
                    continue

                if line == "":  # end of one SSE event
                    if not buf:
                        continue
                    data = "\n".join(buf)
                    buf.clear()
                    if data == "[DONE]":
                        return
                    yield from _parse_sse_event(data)
                # ignore other SSE field lines (event:, id:, comments)

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        payload = self._build_payload(messages, tools, capture_thinking, stream=False, kwargs=kwargs)
        with self._clarify_timeouts(stream=False):
            async with httpx.AsyncClient(timeout=self._httpx_timeout(stream=False)) as client:
                response = await client.post(_API_URL, headers=self._headers(stream=False), json=payload)
        return _turn_from_blocking_json(response.json(), capture_thinking=capture_thinking)

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        payload = self._build_payload(messages, tools, capture_thinking, stream=True, kwargs=kwargs)

        with self._clarify_timeouts(stream=True):
            async with httpx.AsyncClient(timeout=self._httpx_timeout(stream=True)) as client:
                async with client.stream("POST", _API_URL, headers=self._headers(stream=True), json=payload) as r:
                    r.raise_for_status()
                    buf: list[str] = []
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            buf.append(line[5:].lstrip())
                            continue
                        if line == "":
                            if not buf:
                                continue
                            data = "\n".join(buf)
                            buf.clear()
                            if data == "[DONE]":
                                return
                            for raw in _parse_sse_event(data):
                                yield raw


def _with_message(e: requests.exceptions.RequestException | httpx.RequestError, message: str) -> Exception:
    """Rebuild a requests/httpx exception as the same type with a new message,
    keeping the attached request (and response, for requests)."""
    if isinstance(e, requests.exceptions.RequestException):
        return type(e)(message, request=e.request, response=e.response)
    try:
        request = e.request
    except RuntimeError:  # httpx raises when no request was attached
        request = None
    return type(e)(message, request=request)


def _turn_from_blocking_json(data: Any, *, capture_thinking: bool) -> _RawTurn:
    """Translate a non-streaming OpenRouter response JSON dict into a `_RawTurn`."""
    data = cast(OpenRouterResponse | OpenRouterResponseError, data)
    if 'error' in data:
        raise ValueError(f"Error from OpenRouter: {data}")
    try:
        usage = _usage_from_wire(data['usage'])
        message = data['choices'][0]['message']
        content = message.get('content', '') or ''
        raw_tcs = message.get('tool_calls') or []
        tool_calls = [TokiToolCall.from_dict(tc) for tc in raw_tcs]
        thought = _extract_reasoning_text(message) if capture_thinking else ''
        return _RawTurn(content=content, tool_calls=tool_calls, thought=thought, usage=usage)
    except KeyError as e:
        raise ValueError(f"Unexpected response format: '{data}'. {e}") from e


def _parse_sse_event(data: str) -> Iterator[_RawChunk]:
    """Parse a single complete SSE event's data payload into zero-or-more `_RawChunk`s.

    The caller is responsible for handling the special `[DONE]` sentinel (which
    terminates the stream) before invoking this. Malformed JSON is silently
    skipped to match the prior sync behavior.
    """
    try:
        obj = cast(OpenRouterResponseDelta | OpenRouterResponseError, json.loads(data))
    except json.JSONDecodeError:
        return
    if 'error' in obj:
        raise ValueError(f"Error from OpenRouter: {obj}")
    try:
        delta = obj["choices"][0]["delta"]
    except (KeyError, IndexError):
        delta = None

    if delta is not None:
        reasoning_text = _extract_reasoning_text(delta)
        if reasoning_text:
            yield _RawThoughtChunk(text=reasoning_text)

        content = delta.get("content")
        if content:
            yield _RawContentChunk(text=content)

        for tc_delta in delta.get("tool_calls") or []:
            yield _tool_call_chunk_from_delta(tc_delta)

    if obj.get("usage"):
        yield _RawUsage(usage=_usage_from_wire(obj["usage"]))  # type: ignore[index]


def _usage_from_wire(raw: dict) -> TokiUsageMetadata:
    """Build a `TokiUsageMetadata` from an OpenRouter `usage` object (the wire shape is OpenAI's, plus cache writes)."""
    prompt_tokens = raw.get("prompt_tokens") or 0
    completion_tokens = raw.get("completion_tokens") or 0
    details = raw.get("prompt_tokens_details") or {}
    return TokiUsageMetadata(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=raw.get("total_tokens") or prompt_tokens + completion_tokens,
        cache_read_tokens=details.get("cached_tokens") or 0,
        cache_write_tokens=details.get("cache_write_tokens") or 0,
    )


def _tool_call_chunk_from_delta(tc_delta: dict) -> _RawToolCallChunk:
    """Translate one OpenRouter tool-call delta entry into a `_RawToolCallChunk`."""
    index = tc_delta.get("index", 0)
    id_ = tc_delta.get("id")
    fn = tc_delta.get("function") or {}
    name = fn.get("name")
    arguments_fragment = fn.get("arguments")
    return _RawToolCallChunk(
        index=index,
        id=id_,
        name=name,
        arguments_fragment=arguments_fragment if arguments_fragment else None,
    )


def _openrouter_offline_count(model: str, wire_messages: list[dict], wire_tools: list[dict] | None) -> int:
    """Heuristic offline token count for an OpenRouter call. Routed through
    `litellm.token_counter` keyed off the upstream model id (e.g.
    `'anthropic/claude-haiku-4-5'`); raises `ImportError` if litellm is not
    importable so the caller can guide the user to `toki[litellm]`.
    """
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "OpenRouterModel.count_tokens(kind='offline') requires litellm. "
            "Install with `pip install toki[litellm]` (or `toki[all]`)."
        ) from e
    return litellm.token_counter(model=model, messages=wire_messages, tools=wire_tools)


def _build_count_payload(model: str, wire_messages: list[dict], wire_tools: list[dict] | None) -> dict:
    """Minimal `/chat/completions` payload that asks for one output token, used
    purely to read `usage.prompt_tokens` off the response."""
    payload: dict = {
        "model": model,
        "messages": wire_messages,
        "max_tokens": 1,
    }
    if wire_tools:
        payload["tools"] = wire_tools
    return payload


def _prompt_tokens_from_count_response(data: Any) -> int:
    if isinstance(data, dict) and 'error' in data:
        raise ValueError(f"Error from OpenRouter: {data}")
    usage = (data or {}).get('usage') or {}
    pt = usage.get('prompt_tokens')
    if pt is None:
        raise RuntimeError(f"OpenRouter response did not include usage.prompt_tokens: {data!r}")
    return int(pt)


# TODO: make wrapper class around OpenRouterModel that interfaces with tools, but as strings rather than via the openrouter API
#       basically for cases where the model either doesn't support tools, or it does but the interface is flaky
#       it should be usable as a drop-in replacement for OpenRouterModel (e.g. in Agent/etc.)
