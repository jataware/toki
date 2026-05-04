import asyncio
import json
import threading
from typing import Any, AsyncIterator, Iterator, Literal

import litellm

from ..model import (
    BaseModel,
    TokenCountEstimate,
    TokiMessage,
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


# Anthropic with extended thinking + tool use requires `thinking_blocks` to round-trip
# on the assistant message. TokiMessage doesn't carry those blocks, so we let litellm
# auto-drop the `thinking` param when they're missing rather than 400-ing.
# See https://docs.litellm.ai/docs/reasoning_content
litellm.modify_params = True

# Route litellm's async path through httpx instead of its default aiohttp transport.
# Aiohttp is marginally faster at very high concurrency but its cached `ClientSession`
# is bound to the event loop it was first created on, which causes
# `RuntimeError: Event loop is closed` (or post-exit "Fatal error on SSL transport"
# spam) for any caller doing multiple `asyncio.run(...)` invocations or just exiting
# after async use. httpx is what the sync path already uses; this gives transport
# parity and clean shutdown. Power users doing high-concurrency batch work can
# re-enable aiohttp by setting `litellm.disable_aiohttp_transport = False` after
# importing toki.
litellm.disable_aiohttp_transport = True


# Server-side reasoning compute knob. Provider-supported subsets vary; the union below
# covers every value any backend currently exposes. Pass `None` (the Python default) to
# disable reasoning entirely — there is no string `'none'`.
ReasoningEffort = Literal['minimal', 'low', 'medium', 'high', 'xhigh']


def _tool_call_to_wire(tc: TokiToolCall) -> dict:
    """OpenAI/litellm wire shape for an assistant tool call. `arguments` is a JSON string."""
    return {
        "id": tc.id,
        "type": tc.type,
        "function": {"name": tc.function.name, "arguments": json.dumps(tc.function.arguments)},
    }


def _msg_to_wire(m: TokiMessage) -> dict:
    out: dict = {"role": m.role, "content": m.content}
    if m.tool_calls is not None:
        out["tool_calls"] = [_tool_call_to_wire(tc) for tc in m.tool_calls]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """litellm's response objects are pydantic-ish but sometimes leak dicts; tolerate both."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class _LiteLLMModel(BaseModel):
    """Shared base for litellm-backed frontends (`OpenAIModel`, `AnthropicModel`, `GoogleModel`).

    Subclasses fix a litellm wire model (e.g. `'openai/gpt-4o'`) and forward the
    api key through. Not user-facing on its own.
    """

    def __init__(
        self,
        *,
        wire_model: str,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
    ):
        super().__init__()
        self._wire_model = wire_model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    def _prepare_call(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        kwargs: dict,
    ) -> tuple[list[dict], dict]:
        """Subclass hook: produce final wire `messages` and merged call kwargs.

        Default impl serializes each `TokiMessage` and merges `tools` /
        `reasoning_effort` into the kwargs dict. Subclasses override to inject
        cache_control markers, manage explicit caches, or otherwise transform
        the call before it hits litellm.
        """
        out_kwargs: dict = dict(kwargs)
        if tools:
            out_kwargs["tools"] = tools
            out_kwargs.setdefault("parallel_tool_calls", self.allow_parallel_tool_calls)
        if self.reasoning_effort is not None and "reasoning_effort" not in out_kwargs and "thinking" not in out_kwargs:
            out_kwargs["reasoning_effort"] = self.reasoning_effort
        wire_messages = [_msg_to_wire(m) for m in messages]
        return wire_messages, out_kwargs

    async def _aprepare_call(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        kwargs: dict,
    ) -> tuple[list[dict], dict]:
        """Async sibling of `_prepare_call`. Default impl just delegates to the
        sync version; subclasses with async-only side effects (e.g. native
        Google cache creation via `client.aio.caches.create`) override this."""
        return self._prepare_call(messages, tools, kwargs)

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        wire_messages, call_kwargs = self._prepare_call(messages, tools, kwargs)
        response = litellm.completion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=wire_messages,
            stream=False,
            **call_kwargs,
        )
        return _build_turn_from_response(response, capture_thinking=capture_thinking)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        wire_messages, call_kwargs = self._prepare_call(messages, tools, kwargs)
        stream = litellm.completion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=wire_messages,
            stream=True,
            **call_kwargs,
        )
        for chunk in stream:
            yield from _translate_streaming_chunk(chunk)

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        wire_messages, call_kwargs = await self._aprepare_call(messages, tools, kwargs)
        response = await litellm.acompletion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=wire_messages,
            stream=False,
            **call_kwargs,
        )
        return _build_turn_from_response(response, capture_thinking=capture_thinking)

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        wire_messages, call_kwargs = await self._aprepare_call(messages, tools, kwargs)
        stream = await litellm.acompletion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=wire_messages,
            stream=True,
            **call_kwargs,
        )
        async for chunk in stream:
            for raw in _translate_streaming_chunk(chunk):
                yield raw

    # ----- token counting ---------------------------------------------------

    def _normalize_for_count(
        self,
        messages: list[TokiMessage | dict],
        tools: ToolsArg,
    ) -> tuple[list[dict], list[dict] | None]:
        """Normalize messages + tools into the OpenAI-style wire shape that
        `litellm.token_counter` and friends expect."""
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, _ = _unwrap_tools(tools)
        wire_messages = [_msg_to_wire(m) for m in normalized]
        return wire_messages, wire_tools

    def _litellm_offline_count(
        self,
        wire_messages: list[dict],
        wire_tools: list[dict] | None,
    ) -> int:
        """Run `litellm.token_counter` (no network) over the wire prompt."""
        return litellm.token_counter(
            model=self._wire_model,
            messages=wire_messages,
            tools=wire_tools,
        )

    @staticmethod
    def _wrap_estimate(raw: int, safety_factor: float) -> TokenCountEstimate:
        return TokenCountEstimate(
            prompt_tokens=round(raw * safety_factor),
            raw_prompt_tokens=raw,
            safety_factor=safety_factor,
        )


def _run_async_blocking(coro_factory) -> Any:
    """Run an async coroutine to completion synchronously, regardless of whether
    the caller is already inside an event loop. Always uses a fresh thread + loop:
    litellm caches httpx clients per-loop, so stuffing a new `asyncio.run` into
    a foreign loop is a footgun.

    `coro_factory` is a zero-arg callable that returns the coroutine — needed
    because the coroutine has to be created on the worker thread's loop.
    """
    holder: dict = {}

    def runner() -> None:
        try:
            holder['ok'] = asyncio.run(coro_factory())
        except BaseException as e:
            holder['err'] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if 'err' in holder:
        raise holder['err']
    return holder['ok']


def _build_turn_from_response(response: Any, *, capture_thinking: bool) -> _RawTurn:
    """Translate a non-streaming litellm completion response into a `_RawTurn`."""
    choice = response.choices[0]
    msg = choice.message
    content = _attr(msg, "content", "") or ""
    raw_tcs = _attr(msg, "tool_calls", None) or []
    tool_calls = [TokiToolCall.from_dict(_normalize_tool_call(tc)) for tc in raw_tcs]
    thought = (_attr(msg, "reasoning_content", "") or "") if capture_thinking else ""
    usage = _normalize_usage(_attr(response, "usage", None))
    return _RawTurn(content=content, tool_calls=tool_calls, thought=thought, usage=usage)


def _translate_streaming_chunk(chunk: Any) -> Iterator[_RawChunk]:
    """Translate one streaming-completion chunk into zero-or-more `_RawChunk`s.
    Shared between the sync and async streaming paths."""
    choices = _attr(chunk, "choices", None) or []
    if choices:
        delta = _attr(choices[0], "delta", None)
        if delta is not None:
            reasoning = _attr(delta, "reasoning_content", None)
            if reasoning:
                yield _RawThoughtChunk(text=reasoning)
            content = _attr(delta, "content", None)
            if content:
                yield _RawContentChunk(text=content)
            tool_deltas = _attr(delta, "tool_calls", None) or []
            for tc in tool_deltas:
                yield _tool_call_chunk_from_delta(tc)
    usage = _attr(chunk, "usage", None)
    if usage is not None:
        normalized = _normalize_usage(usage)
        if normalized is not None:
            yield _RawUsage(usage=normalized)


def _normalize_tool_call(tc: Any) -> dict:
    """Coerce a litellm tool-call (pydantic or dict) into the dict shape `TokiToolCall.from_dict` expects."""
    fn = _attr(tc, "function", None) or {}
    return {
        "id": _attr(tc, "id", "") or "",
        "type": _attr(tc, "type", "function") or "function",
        "function": {
            "name": _attr(fn, "name", "") or "",
            "arguments": _attr(fn, "arguments", "") or "",
        },
    }


def _tool_call_chunk_from_delta(tc_delta: Any) -> _RawToolCallChunk:
    fn = _attr(tc_delta, "function", None) or {}
    args = _attr(fn, "arguments", None)
    return _RawToolCallChunk(
        index=_attr(tc_delta, "index", 0) or 0,
        id=_attr(tc_delta, "id", None) or None,
        name=_attr(fn, "name", None) or None,
        arguments_fragment=args if args else None,
    )


def _normalize_usage(u: Any) -> TokiUsageMetadata | None:
    if u is None:
        return None
    pt = _attr(u, "prompt_tokens", None)
    ct = _attr(u, "completion_tokens", None)
    tt = _attr(u, "total_tokens", None)
    if pt is None and ct is None and tt is None:
        return None
    return TokiUsageMetadata(
        prompt_tokens=pt or 0,
        completion_tokens=ct or 0,
        total_tokens=tt if tt is not None else (pt or 0) + (ct or 0),
    )
