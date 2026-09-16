"""TokiMessage / ToolSchema ↔ OpenAI Responses input items.

Pure conversion: no SDK import, so unit tests can run without `openai`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..model import (
    TokiMessage,
    TokiToolCall,
    TokiToolFunction,
    TokiUsageMetadata,
    _RawChunk,
    _RawContentChunk,
    _RawProviderState,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
)

PROVIDER_STATE_KEY = "openai_responses"


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def item_to_dict(item: Any) -> dict:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return {k: getattr(item, k) for k in dir(item) if not k.startswith("_")}


def flatten_tools(wire_tools: list[dict] | None) -> list[dict] | None:
    """Chat Completions `{type, function:{name, description, parameters}}` → Responses function tools."""
    if not wire_tools:
        return None
    out: list[dict] = []
    for tool in wire_tools:
        fn = tool.get("function", tool)
        item: dict = {
            "type": "function",
            "name": fn["name"],
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            item["description"] = fn["description"]
        out.append(item)
    return out


def reasoning_items_from_message(message: TokiMessage) -> list[dict]:
    if message.provider_state is None:
        return []
    state = message.provider_state.get(PROVIDER_STATE_KEY)
    if state is None:
        return []
    return list(state.get("reasoning_items") or [])


def messages_to_input(messages: list[TokiMessage]) -> list[dict]:
    items: list[dict] = []
    for message in messages:
        if message.role in ("system", "user"):
            items.append({
                "type": "message",
                "role": message.role,
                "content": message.content,
            })
            continue
        if message.role == "assistant":
            items.extend(reasoning_items_from_message(message))
            if message.content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": message.content,
                })
            for tc in message.tool_calls or []:
                args = tc.function.arguments
                if not isinstance(args, str):
                    args = json.dumps(args)
                items.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
            continue
        items.append({
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        })
    return items


def _message_text(item: Any) -> str:
    content = _attr(item, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if _attr(block, "type") in ("output_text", "input_text", "text"):
            parts.append(_attr(block, "text") or "")
    return "".join(parts)


def _reasoning_summary_text(item: Any) -> str:
    summary = _attr(item, "summary") or []
    parts: list[str] = []
    for block in summary:
        parts.append(_attr(block, "text") or "")
    return "".join(parts)


def provider_state_from_output(output: Any) -> dict[str, Any] | None:
    if not output:
        return None
    reasoning_items = [
        item_to_dict(item) for item in output if _attr(item, "type") == "reasoning"
    ]
    if not reasoning_items:
        return None
    return {PROVIDER_STATE_KEY: {"reasoning_items": reasoning_items}}


def usage_from_wire(usage: Any) -> TokiUsageMetadata | None:
    if usage is None:
        return None
    prompt = _attr(usage, "input_tokens", None)
    completion = _attr(usage, "output_tokens", None)
    total = _attr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        prompt = _attr(usage, "prompt_tokens", None)
        completion = _attr(usage, "completion_tokens", None)
        total = _attr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        return None
    details = _attr(usage, "input_tokens_details", None)
    cached = _attr(details, "cached_tokens", 0) or 0 if details is not None else 0
    prompt_n = prompt or 0
    completion_n = completion or 0
    return TokiUsageMetadata(
        prompt_tokens=prompt_n,
        completion_tokens=completion_n,
        total_tokens=total if total is not None else prompt_n + completion_n,
        cache_read_tokens=cached,
    )


def _tool_call_from_item(item: Any) -> TokiToolCall:
    args = _attr(item, "arguments") or "{}"
    return TokiToolCall(
        id=_attr(item, "call_id") or _attr(item, "id") or "",
        function=TokiToolFunction.from_dict({
            "name": _attr(item, "name") or "",
            "arguments": args,
        }),
    )


def turn_from_response(response: Any, *, capture_thinking: bool) -> _RawTurn:
    output = _attr(response, "output") or []
    content_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[TokiToolCall] = []
    for item in output:
        itype = _attr(item, "type")
        if itype == "message":
            content_parts.append(_message_text(item))
        elif itype == "function_call":
            tool_calls.append(_tool_call_from_item(item))
        elif itype == "reasoning" and capture_thinking:
            thought_parts.append(_reasoning_summary_text(item))
    return _RawTurn(
        content="".join(content_parts),
        tool_calls=tool_calls,
        thought="".join(thought_parts) if capture_thinking else "",
        usage=usage_from_wire(_attr(response, "usage")),
        provider_state=provider_state_from_output(output),
    )


class ResponsesStreamDecoder:
    """Translate Responses SSE / SDK stream events into `_RawChunk`s."""

    def __init__(self, *, capture_thinking: bool) -> None:
        self.capture_thinking = capture_thinking
        self._item_index: dict[str, int] = {}

    def feed(self, event: Any) -> Iterator[_RawChunk]:
        etype = _attr(event, "type") or ""
        if etype == "response.output_text.delta":
            delta = _attr(event, "delta") or ""
            if delta:
                yield _RawContentChunk(text=delta)
            return
        if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            if self.capture_thinking:
                delta = _attr(event, "delta") or ""
                if delta:
                    yield _RawThoughtChunk(text=delta)
            return
        if etype == "response.output_item.added":
            yield from self._on_item_added(event)
            return
        if etype == "response.function_call_arguments.delta":
            item_id = _attr(event, "item_id")
            index = _attr(event, "output_index")
            if index is None:
                index = self._item_index.get(item_id, 0)
            delta = _attr(event, "delta") or ""
            if delta:
                yield _RawToolCallChunk(index=index, arguments_fragment=delta)
            return
        if etype == "response.completed":
            response = _attr(event, "response")
            if response is None:
                return
            usage = usage_from_wire(_attr(response, "usage"))
            if usage is not None:
                yield _RawUsage(usage=usage)
            ps = provider_state_from_output(_attr(response, "output"))
            if ps is not None:
                yield _RawProviderState(provider_state=ps)

    def _on_item_added(self, event: Any) -> Iterator[_RawChunk]:
        item = _attr(event, "item")
        index = _attr(event, "output_index")
        if index is None:
            index = 0
        item_id = _attr(item, "id")
        if item_id:
            self._item_index[item_id] = index
        if _attr(item, "type") != "function_call":
            return
        args = _attr(item, "arguments") or ""
        yield _RawToolCallChunk(
            index=index,
            id=_attr(item, "call_id") or item_id,
            name=_attr(item, "name"),
            arguments_fragment=args if args else None,
        )
