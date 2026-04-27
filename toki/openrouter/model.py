import json
import warnings
from typing import Any, Generator, Literal, TypedDict, cast, overload

import requests
from typing_extensions import NotRequired

from ..model import BaseModel, TokiMessage, TokiToolCall, TokiToolResponse, TokiUsageMetadata
from .models import OpenRouterModelName


class OpenRouterResponseCompletionChoice(TypedDict):
    message: TokiMessage


class OpenRouterResponse(TypedDict):
    choices: list[OpenRouterResponseCompletionChoice]
    usage: TokiUsageMetadata


class OpenRouterResponseDeltaPayload(TypedDict):
    content: NotRequired[str]
    tool_calls: NotRequired[list[TokiToolCall]]


class OpenRouterResponseChoice(TypedDict):
    delta: OpenRouterResponseDeltaPayload


class OpenRouterResponseDelta(TypedDict):
    choices: list[OpenRouterResponseChoice]
    usage: NotRequired[TokiUsageMetadata]  # typically only on the final chunk


class OpenRouterResponseError(TypedDict):
    error: Any


class OpenRouterModel(BaseModel):
    """Toki model backend that talks to OpenRouter's chat-completions API over HTTPS."""

    def __init__(self, model: OpenRouterModelName, api_key: str, allow_parallel_tool_calls: bool = False, cache: bool = False):
        super().__init__()
        if cache:
            warnings.warn("cache=True is not yet implemented; ignoring", stacklevel=2)
        self.model = model
        self.api_key = api_key
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> str | TokiToolResponse: ...
    def _blocking_complete(self, messages: list[TokiMessage], tools: list | None = None, **kwargs) -> str | TokiToolResponse:
        tool_payload = {"tools": tools, "parallel_tool_calls": self.allow_parallel_tool_calls} if tools else {}
        payload = {"model": self.model, "messages": messages, **tool_payload, **kwargs}
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload
        )
        data = cast(OpenRouterResponse | OpenRouterResponseError, response.json())
        if 'error' in data:
            raise ValueError(f"Error from OpenRouter: {data}")
        try:
            self._usage_metadata = cast(TokiUsageMetadata, data['usage'])
            if 'tool_calls' in data['choices'][0]['message'] and len(data['choices'][0]['message']['tool_calls']) > 0:
                return TokiToolResponse(thought=data['choices'][0]['message']['content'], tool_calls=data['choices'][0]['message']['tool_calls'])
            return data['choices'][0]['message']['content']
        except KeyError as e:
            raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
        except Exception as e:
            raise ValueError(f"An error occurred while processing the response: '{data}'. {e}") from e

    # TODO: should request timeout be a setting rather than hardcoded?
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    def _streaming_complete(self, messages: list[TokiMessage], tools: list | None = None, **kwargs) -> Generator[str | TokiToolResponse, None, None]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        tool_payload = {"tools": tools, "parallel_tool_calls": self.allow_parallel_tool_calls} if tools else {}
        payload = {"model": self.model, "messages": messages, "stream": True, **tool_payload, **kwargs}

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

                    # parse the chunk and yield any content
                    try:
                        obj = cast(OpenRouterResponseDelta | OpenRouterResponseError, json.loads(data))
                    except json.JSONDecodeError:
                        continue  # wait for the next complete event
                    if 'error' in obj:
                        raise ValueError(f"Error from OpenRouter: {obj}")
                    try:
                        content: str | None = obj["choices"][0]["delta"].get("content")
                        tool_calls: list[TokiToolCall] | None = obj["choices"][0]["delta"].get("tool_calls")
                        if tool_calls:
                            yield TokiToolResponse(thought=content or '', tool_calls=tool_calls)
                        elif content:
                            yield content
                    except KeyError as e:
                        raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e

                    # update the usage metadata (typically on the final chunk)
                    if "usage" in obj:
                        self._usage_metadata = cast(TokiUsageMetadata, obj["usage"])  # type: ignore[index]

                    continue

                # ignore other fields like "event:" / "id:" / comments / etc.


# TODO: make wrapper class around OpenRouterModel that interfaces with tools, but as strings rather than via the openrouter API
#       basically for cases where the model either doesn't support tools, or it does but the interface is flaky
#       it should be usable as a drop-in replacement for OpenRouterModel (e.g. in Agent/etc.)
