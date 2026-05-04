from typing import Literal

import litellm

from ..helpers.cache_state import _CacheState, estimate_messages_tokens
from ..litellm.model import ReasoningEffort, _LiteLLMModel, _run_async_blocking
from ..model import TokenCountEstimate, TokiMessage, ToolsArg
from .models import AnthropicModelName
from .utils import apply_cache_markers


_ANTHROPIC_OFFLINE_SAFETY_FACTOR_DEFAULT = 1.15


# Anthropic's documented minimum cacheable prompt is 1024 tokens for most
# models. Our `chars/4` estimate is conservative enough that we use the same
# floor here.
_ANTHROPIC_MIN_CACHE_TOKENS = 1024


class AnthropicModel(_LiteLLMModel):
    """Toki frontend for Anthropic's Claude models, dispatched via litellm.

    Caching is opt-in via `cache=`:

      - `cache='rolling'` — every turn marks the system prompt, the last tool
        definition, and the most-recent message with `cache_control`. The
        cache breakpoint advances automatically as the conversation grows;
        each turn writes its tail and reads the entire prior history.
      - `cache='static'` — anchors the cache breakpoint at a fixed point in
        history. `_CacheState` defers the snapshot until the prefix is large
        enough to actually be cached (>= 1024 tokens by `chars/4` estimate);
        from then on, every call places markers on system + last tool + the
        pinned boundary message.
      - `cache=None` (default) — no markers; behaves exactly like a
        cache-unaware request.

    `model.cache` is a regular mutable attribute, so a session can move
    between strategies turn-to-turn. `model.invalidate_cache()` drops the
    historical anchor list so the next `'static'` call snapshots a fresh
    anchor.
    """

    def __init__(
        self,
        model: AnthropicModelName | str,
        *,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
        cache: Literal['rolling', 'static'] | None = None,
        cache_ttl: Literal['5m', '1h'] = '5m',
    ):
        super().__init__(
            wire_model=f"anthropic/{model}",
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )
        self.model = model
        self.cache = cache
        self.cache_ttl = cache_ttl
        self._cache_state = _CacheState(min_cache_size_estimate=_ANTHROPIC_MIN_CACHE_TOKENS)

    def invalidate_cache(self) -> None:
        """Drop the historical anchor list. Next `'static'` call will defer
        until the prefix is large enough and then anchor afresh."""
        self._cache_state.clear()

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _ANTHROPIC_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"AnthropicModel.count_tokens: unsupported kind {kind!r}")
        wire_messages, wire_tools = self._normalize_for_count(messages, tools)
        if kind == 'offline':
            raw = self._litellm_offline_count(wire_messages, wire_tools)
            return self._wrap_estimate(raw, safety_factor)
        # exact / online both hit Anthropic's count-tokens endpoint via litellm
        response = _run_async_blocking(lambda: litellm.acount_tokens(
            model=self._wire_model,
            messages=wire_messages,
            tools=wire_tools,
            api_key=self.api_key,
        ))
        return response.total_tokens

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact', 'offline', 'online'] = 'exact',
        safety_factor: float = _ANTHROPIC_OFFLINE_SAFETY_FACTOR_DEFAULT,
    ) -> int | TokenCountEstimate:
        if kind not in ('exact', 'offline', 'online'):
            raise ValueError(f"AnthropicModel.acount_tokens: unsupported kind {kind!r}")
        wire_messages, wire_tools = self._normalize_for_count(messages, tools)
        if kind == 'offline':
            raw = self._litellm_offline_count(wire_messages, wire_tools)
            return self._wrap_estimate(raw, safety_factor)
        response = await litellm.acount_tokens(
            model=self._wire_model,
            messages=wire_messages,
            tools=wire_tools,
            api_key=self.api_key,
        )
        return response.total_tokens

    def _prepare_call(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        kwargs: dict,
    ) -> tuple[list[dict], dict]:
        wire_messages, out_kwargs = super()._prepare_call(messages, tools, kwargs)
        if self.cache is None:
            return wire_messages, out_kwargs

        if self.cache == 'rolling':
            anchor_index = len(messages)
        else:
            entry = self._cache_state.match_or_snapshot(
                strategy='static',
                messages=messages,
                system=None,
                tools=tools,
                prefix_token_estimate=estimate_messages_tokens(None, tools, messages),
            )
            if entry is None:
                return wire_messages, out_kwargs
            anchor_index = entry.anchor_index

        wire_tools = out_kwargs.get("tools")
        new_wire_messages, new_wire_tools = apply_cache_markers(
            wire_messages,
            wire_tools,
            mode=self.cache,
            anchor_index=anchor_index,
            ttl=self.cache_ttl,
        )
        if new_wire_tools is not None:
            out_kwargs["tools"] = new_wire_tools
        return new_wire_messages, out_kwargs
