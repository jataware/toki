import json
import warnings
from typing import Any, AsyncIterator, Iterator, TypedDict, cast

import httpx
import requests
from typing_extensions import NotRequired

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
from .models import OpenRouterModelName


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
    usage: TokiUsageMetadata


class OpenRouterResponseDeltaPayload(TypedDict):
    content: NotRequired[str]
    tool_calls: NotRequired[list[dict]]
    reasoning: NotRequired[str]
    reasoning_details: NotRequired[list[OpenRouterReasoningDetail]]


class OpenRouterResponseChoice(TypedDict):
    delta: OpenRouterResponseDeltaPayload


class OpenRouterResponseDelta(TypedDict):
    choices: list[OpenRouterResponseChoice]
    usage: NotRequired[TokiUsageMetadata]  # typically only on the final chunk


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
    """Toki model backend that talks to OpenRouter's chat-completions API over HTTPS."""

    def __init__(self, model: OpenRouterModelName, api_key: str, allow_parallel_tool_calls: bool = False, cache: bool = False):
        super().__init__()
        if cache:
            warnings.warn("cache=True is not yet implemented; ignoring", stacklevel=2)
        self.model = model
        self.api_key = api_key
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    def _build_payload(self, messages: list[TokiMessage], tools: list[dict] | None, capture_thinking: bool, stream: bool, kwargs: dict) -> dict:
        tool_payload = {"tools": tools, "parallel_tool_calls": self.allow_parallel_tool_calls} if tools else {}
        reasoning_payload = {"reasoning": {"enabled": True}} if capture_thinking else {}
        payload: dict = {
            "model": self.model,
            "messages": [_msg_to_wire(m) for m in messages],
            **tool_payload,
            **reasoning_payload,
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

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        payload = self._build_payload(messages, tools, capture_thinking, stream=False, kwargs=kwargs)
        response = requests.post(url=_API_URL, headers=self._headers(stream=False), json=payload)
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

        with requests.post(_API_URL, headers=self._headers(stream=True), json=payload, stream=True, timeout=(10, 60)) as r:
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
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

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
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


def _turn_from_blocking_json(data: Any, *, capture_thinking: bool) -> _RawTurn:
    """Translate a non-streaming OpenRouter response JSON dict into a `_RawTurn`."""
    data = cast(OpenRouterResponse | OpenRouterResponseError, data)
    if 'error' in data:
        raise ValueError(f"Error from OpenRouter: {data}")
    try:
        usage = cast(TokiUsageMetadata, data['usage'])
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

    if "usage" in obj:
        yield _RawUsage(usage=cast(TokiUsageMetadata, obj["usage"]))  # type: ignore[index]


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


# TODO: make wrapper class around OpenRouterModel that interfaces with tools, but as strings rather than via the openrouter API
#       basically for cases where the model either doesn't support tools, or it does but the interface is flaky
#       it should be usable as a drop-in replacement for OpenRouterModel (e.g. in Agent/etc.)
