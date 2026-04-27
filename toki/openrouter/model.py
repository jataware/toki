import json
import warnings
from dataclasses import asdict
from typing import Any, Generator, Literal, TypedDict, cast, overload

import requests
from typing_extensions import NotRequired

from ..model import (
    BaseModel,
    TokiChatResponse,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolResponse,
    TokiUsageMetadata,
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


def _msg_to_wire(m: TokiMessage) -> dict:
    """Serialize a TokiMessage dataclass to the OpenRouter wire shape, dropping None fields."""
    out: dict = {"role": m.role, "content": m.content}
    if m.tool_calls is not None:
        out["tool_calls"] = [asdict(tc) for tc in m.tool_calls]
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

    def _build_payload(self, messages: list[TokiMessage], tools: list | None, capture_thinking: bool, stream: bool, kwargs: dict) -> dict:
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

    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse | TokiToolResponse: ...
    def _blocking_complete(self, messages: list[TokiMessage], tools: list | None = None, *, capture_thinking: bool = False, **kwargs) -> str | TokiChatResponse | TokiToolResponse:
        payload = self._build_payload(messages, tools, capture_thinking, stream=False, kwargs=kwargs)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        data = cast(OpenRouterResponse | OpenRouterResponseError, response.json())
        if 'error' in data:
            raise ValueError(f"Error from OpenRouter: {data}")
        try:
            self._usage_metadata = cast(TokiUsageMetadata, data['usage'])
            message = data['choices'][0]['message']
            content = message.get('content', '') or ''
            raw_tcs = message.get('tool_calls')
            if raw_tcs:
                tool_calls = [TokiToolCall.from_dict(tc) for tc in raw_tcs]
                thought = _extract_reasoning_text(message) if capture_thinking else content
                return TokiToolResponse(thought=thought, tool_calls=tool_calls)
            if capture_thinking:
                return TokiChatResponse(content=content, thought=_extract_reasoning_text(message))
            return content
        except KeyError as e:
            raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
        except Exception as e:
            raise ValueError(f"An error occurred while processing the response: '{data}'. {e}") from e

    # TODO: should request timeout be a setting rather than hardcoded?
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    def _streaming_complete(self, messages: list[TokiMessage], tools: list | None = None, *, capture_thinking: bool = False, **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(messages, tools, capture_thinking, stream=True, kwargs=kwargs)

        accumulated_reasoning: list[str] = []

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            buf = []
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

                    try:
                        obj = cast(OpenRouterResponseDelta | OpenRouterResponseError, json.loads(data))
                    except json.JSONDecodeError:
                        continue
                    if 'error' in obj:
                        raise ValueError(f"Error from OpenRouter: {obj}")
                    try:
                        delta = obj["choices"][0]["delta"]
                        reasoning_text = _extract_reasoning_text(delta)
                        if reasoning_text:
                            accumulated_reasoning.append(reasoning_text)
                            if capture_thinking:
                                yield TokiThinking(text=reasoning_text)

                        content: str | None = delta.get("content")
                        raw_tcs = delta.get("tool_calls")
                        if raw_tcs:
                            tool_calls = [TokiToolCall.from_dict(tc) for tc in raw_tcs]
                            thought = ''.join(accumulated_reasoning) if capture_thinking else (content or '')
                            yield TokiToolResponse(thought=thought, tool_calls=tool_calls)
                        elif content:
                            yield content
                    except KeyError as e:
                        raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e

                    if "usage" in obj:
                        self._usage_metadata = cast(TokiUsageMetadata, obj["usage"])  # type: ignore[index]

                    continue

                # ignore other fields like "event:" / "id:" / comments / etc.


# TODO: make wrapper class around OpenRouterModel that interfaces with tools, but as strings rather than via the openrouter API
#       basically for cases where the model either doesn't support tools, or it does but the interface is flaky
#       it should be usable as a drop-in replacement for OpenRouterModel (e.g. in Agent/etc.)
