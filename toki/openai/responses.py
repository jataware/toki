"""Toki frontend for OpenAI's Responses API.

`OpenAIModel` stays on Chat Completions via litellm. This backend exists so
tools and `reasoning_effort` work together (the GPT-5.4 Completions hole).
History stays on `TokiMessage` lists; Responses `store` is always off.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Literal

from openai import AsyncOpenAI, OpenAI

from ..model import (
    BaseModel,
    ReasoningEffort,
    TokiMessage,
    ToolsArg,
    _RawChunk,
    _RawTurn,
    _unwrap_tools,
)
from .models import OpenAIModelName, attributes_map
from .responses_wire import (
    ResponsesStreamDecoder,
    flatten_tools,
    messages_to_input,
    turn_from_response,
)


class OpenAIResponsesModel(BaseModel):
    """Toki frontend for OpenAI's Responses API.

    Model ids match OpenAI's catalog (same `OpenAIModelName` as `OpenAIModel`).
    Each call converts `TokiMessage` history to Responses `input` items and
    collapses output back to toki types. Always sends `store=False` — conversation
    state stays on `Agent.messages`. Reasoning items round-trip through
    `provider_state['openai_responses']` so a tool turn can replay them.

    No `cache=` kwarg: OpenAI's prompt-prefix cache is automatic for prompts
    >= 1024 tokens and cannot be disabled or controlled.

    Extra: `toki[openai-responses]` (official `openai` SDK + tiktoken).
    `toki[openai]` is the Chat Completions / litellm frontend (`OpenAIModel`).
    """

    def __init__(
        self,
        model: OpenAIModelName | str,
        *,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    def _get_allow_parallel_tool_calls(self) -> bool:
        return self.allow_parallel_tool_calls

    def _supports_thinking(self) -> bool | None:
        attr = attributes_map.get(self.model)
        if attr is None:
            return None
        return getattr(attr, "supports_thinking", None)

    def _create_kwargs(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        kwargs: dict,
        *,
        capture_thinking: bool,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "input": messages_to_input(messages),
            "store": False,
        }
        flat_tools = flatten_tools(tools)
        if flat_tools:
            payload["tools"] = flat_tools
            payload["parallel_tool_calls"] = self.allow_parallel_tool_calls
        if "reasoning" not in kwargs:
            effort = self.reasoning_effort
            if effort is None and capture_thinking:
                effort = "medium"
            if effort is not None:
                payload["reasoning"] = {"effort": effort, "summary": "auto"}
                payload["include"] = ["reasoning.encrypted_content"]
        payload.update(kwargs)
        payload["store"] = False
        return payload

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        response = self._client.responses.create(
            **self._create_kwargs(messages, tools, kwargs, capture_thinking=capture_thinking),
        )
        return turn_from_response(response, capture_thinking=capture_thinking)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        stream = self._client.responses.create(
            **self._create_kwargs(messages, tools, kwargs, capture_thinking=capture_thinking),
            stream=True,
        )
        decoder = ResponsesStreamDecoder(capture_thinking=capture_thinking)
        for event in stream:
            yield from decoder.feed(event)

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        response = await self._async_client.responses.create(
            **self._create_kwargs(messages, tools, kwargs, capture_thinking=capture_thinking),
        )
        return turn_from_response(response, capture_thinking=capture_thinking)

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        stream = await self._async_client.responses.create(
            **self._create_kwargs(messages, tools, kwargs, capture_thinking=capture_thinking),
            stream=True,
        )
        decoder = ResponsesStreamDecoder(capture_thinking=capture_thinking)
        async for event in stream:
            for raw in decoder.feed(event):
                yield raw

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal["exact"] = "exact",
    ) -> int:
        """Count prompt tokens for the converted Responses `input` (and tools).

        Offline via tiktoken over the JSON of the Responses payload. Same
        tokenizer family as Chat Completions (`o200k_base` when the model id
        is unknown to tiktoken). Only `kind='exact'` is exposed.
        """
        if kind != "exact":
            raise ValueError(f"OpenAIResponsesModel only supports kind='exact'; got {kind!r}")
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, _ = _unwrap_tools(tools)
        blob = json.dumps(
            {
                "input": messages_to_input(normalized),
                "tools": flatten_tools(wire_tools) or [],
            },
            ensure_ascii=False,
        )
        return _tiktoken_len(self.model, blob)


def _tiktoken_len(model: str, text: str) -> int:
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))
