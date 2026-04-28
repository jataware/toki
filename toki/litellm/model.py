import json
import warnings
from typing import Any, Iterator, Literal

import litellm

from ..model import (
    BaseModel,
    TokiMessage,
    TokiToolCall,
    TokiUsageMetadata,
    _RawChunk,
    _RawContentChunk,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
)


# Anthropic with extended thinking + tool use requires `thinking_blocks` to round-trip
# on the assistant message. TokiMessage doesn't carry those blocks, so we let litellm
# auto-drop the `thinking` param when they're missing rather than 400-ing.
# See https://docs.litellm.ai/docs/reasoning_content
litellm.modify_params = True


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
        cache: bool = False,
    ):
        super().__init__()
        if cache:
            warnings.warn("cache=True is not yet implemented; ignoring", stacklevel=2)
        self._wire_model = wire_model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    def _build_kwargs(self, tools: list[dict] | None, capture_thinking: bool, kwargs: dict) -> dict:
        out: dict = dict(kwargs)
        if tools:
            out["tools"] = tools
            out.setdefault("parallel_tool_calls", self.allow_parallel_tool_calls)
        if self.reasoning_effort is not None and "reasoning_effort" not in out and "thinking" not in out:
            out["reasoning_effort"] = self.reasoning_effort
        return out

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        call_kwargs = self._build_kwargs(tools, capture_thinking, kwargs)
        response = litellm.completion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=[_msg_to_wire(m) for m in messages],
            stream=False,
            **call_kwargs,
        )
        choice = response.choices[0]
        msg = choice.message
        content = _attr(msg, "content", "") or ""
        raw_tcs = _attr(msg, "tool_calls", None) or []
        tool_calls = [TokiToolCall.from_dict(_normalize_tool_call(tc)) for tc in raw_tcs]
        thought = (_attr(msg, "reasoning_content", "") or "") if capture_thinking else ""
        usage = _normalize_usage(_attr(response, "usage", None))
        return _RawTurn(content=content, tool_calls=tool_calls, thought=thought, usage=usage)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        call_kwargs = self._build_kwargs(tools, capture_thinking, kwargs)
        stream = litellm.completion(
            model=self._wire_model,
            api_key=self.api_key,
            messages=[_msg_to_wire(m) for m in messages],
            stream=True,
            **call_kwargs,
        )
        for chunk in stream:
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
