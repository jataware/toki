import asyncio
import os
from asyncio import Queue
from collections.abc import AsyncIterator, Iterator
from threading import Event, Thread
from typing import Any, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ..helpers.cache_state import _CacheState, estimate_messages_tokens
from ..model import (
    BaseModel,
    ReasoningEffort,
    TokiBackendQuirkWarning,
    TokiCacheWarning,
    TokiMessage,
    TokiToolCall,
    TokiToolFunction,
    TokiUsageMetadata,
    ToolsArg,
    _RawChunk,
    _RawContentChunk,
    _RawProviderState,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
    _unwrap_tools,
)
from .cache import apply_cache_point
from .counting import count_claude_tokens
from .models import (
    BedrockModelName,
    attributes_map,
    get_bedrock_model_attributes,
)
from .reasoning import (
    BedrockReasoningConfig,
    ClaudeBudgetReasoning,
    default_reasoning_config,
    reasoning_family,
    reasoning_request_fields,
)

_BEDROCK_PROVIDER_STATE_KEY = "bedrock"
_STREAM_END = object()


def _tools_to_wire(tools: list[dict] | None) -> dict | None:
    if not tools:
        return None
    specs: list[dict] = []
    for tool in tools:
        function = tool["function"]
        spec: dict = {
            "name": function["name"],
            "inputSchema": {"json": function["parameters"]},
        }
        if "description" in function:
            spec["description"] = function["description"]
        specs.append({"toolSpec": spec})
    return {"tools": specs}


def _reasoning_blocks(message: TokiMessage) -> list[dict]:
    if message.provider_state is None:
        return []
    bedrock_state = message.provider_state.get(_BEDROCK_PROVIDER_STATE_KEY)
    if bedrock_state is None:
        return []
    return bedrock_state["reasoning_content"]


def _messages_to_wire(
    messages: list[TokiMessage],
) -> tuple[list[dict] | None, list[dict]]:
    system: list[dict] = []
    wire_messages: list[dict] = []
    pending_tool_results: list[dict] | None = None

    for message in messages:
        if message.role == "system":
            system.append({"text": message.content})
            continue

        if message.role == "tool":
            block = {
                "toolResult": {
                    "toolUseId": message.tool_call_id,
                    "content": [{"text": message.content}],
                }
            }
            if pending_tool_results is None:
                pending_tool_results = [block]
                wire_messages.append({"role": "user", "content": pending_tool_results})
            else:
                pending_tool_results.append(block)
            continue

        pending_tool_results = None
        if message.role == "user":
            wire_messages.append(
                {
                    "role": "user",
                    "content": [{"text": message.content}],
                }
            )
            continue

        content = list(_reasoning_blocks(message))
        if message.content:
            content.append({"text": message.content})
        for tool_call in message.tool_calls or []:
            content.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.function.name,
                        "input": tool_call.function.arguments,
                    }
                }
            )
        wire_messages.append({"role": "assistant", "content": content})

    return system or None, wire_messages


def _usage_from_wire(usage: dict | None) -> TokiUsageMetadata | None:
    if usage is None:
        return None
    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
    prompt_tokens = usage.get("inputTokens", 0) + cache_read_tokens + cache_write_tokens
    completion_tokens = usage.get("outputTokens", 0)
    return TokiUsageMetadata(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _provider_state(reasoning_content: list[dict]) -> dict[str, Any] | None:
    if not reasoning_content:
        return None
    return {
        _BEDROCK_PROVIDER_STATE_KEY: {
            "reasoning_content": reasoning_content,
        }
    }


def _turn_from_response(response: dict, *, capture_thinking: bool) -> _RawTurn:
    content_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[TokiToolCall] = []
    reasoning_content: list[dict] = []

    for block in response["output"]["message"]["content"]:
        if "reasoningContent" in block:
            reasoning_content.append({"reasoningContent": block["reasoningContent"]})
            reasoning_text = block["reasoningContent"].get("reasoningText")
            if capture_thinking and reasoning_text is not None:
                thought_parts.append(reasoning_text.get("text", ""))
        elif "text" in block:
            content_parts.append(block["text"])
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append(
                TokiToolCall(
                    id=tool_use["toolUseId"],
                    function=TokiToolFunction(
                        name=tool_use["name"],
                        arguments=tool_use["input"],
                    ),
                )
            )

    if tool_calls:
        tool_calls[0].provider_state = _provider_state(reasoning_content)

    return _RawTurn(
        content="".join(content_parts),
        tool_calls=tool_calls,
        thought="".join(thought_parts),
        usage=_usage_from_wire(response.get("usage")),
        provider_state=_provider_state(reasoning_content),
    )


def _stream_error(event: dict) -> RuntimeError | None:
    for name, detail in event.items():
        if name.endswith("Exception"):
            message = detail.get("message", repr(detail))
            return RuntimeError(f"Bedrock stream {name}: {message}")
    return None


def _stream_chunks(stream: Iterator[dict]) -> Iterator[_RawChunk]:
    reasoning_parts: dict[int, dict] = {}
    reasoning_content: list[dict] = []
    state_attached = False

    try:
        for event in stream:
            error = _stream_error(event)
            if error is not None:
                raise error

            if "contentBlockStart" in event:
                started = event["contentBlockStart"]
                tool_use = started["start"].get("toolUse")
                if tool_use is not None:
                    provider_state = None
                    if not state_attached:
                        provider_state = _provider_state(reasoning_content)
                        state_attached = provider_state is not None
                    yield _RawToolCallChunk(
                        index=started["contentBlockIndex"],
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        provider_state=provider_state,
                    )
                continue

            if "contentBlockDelta" in event:
                changed = event["contentBlockDelta"]
                index = changed["contentBlockIndex"]
                delta = changed["delta"]
                if "text" in delta:
                    yield _RawContentChunk(text=delta["text"])
                if "reasoningContent" in delta:
                    reasoning_delta = delta["reasoningContent"]
                    current = reasoning_parts.setdefault(index, {})
                    if "text" in reasoning_delta:
                        current["text"] = (
                            current.get("text", "") + reasoning_delta["text"]
                        )
                        yield _RawThoughtChunk(text=reasoning_delta["text"])
                    if "signature" in reasoning_delta:
                        current["signature"] = (
                            current.get("signature", "") + reasoning_delta["signature"]
                        )
                    if "redactedContent" in reasoning_delta:
                        current["redactedContent"] = (
                            current.get("redactedContent", b"")
                            + reasoning_delta["redactedContent"]
                        )
                if "toolUse" in delta:
                    yield _RawToolCallChunk(
                        index=index,
                        arguments_fragment=delta["toolUse"].get("input"),
                    )
                continue

            if "contentBlockStop" in event:
                index = event["contentBlockStop"]["contentBlockIndex"]
                reasoning = reasoning_parts.pop(index, None)
                if reasoning is not None:
                    if "redactedContent" in reasoning:
                        block = {"redactedContent": reasoning["redactedContent"]}
                    else:
                        block = {"reasoningText": reasoning}
                    reasoning_content.append({"reasoningContent": block})
                continue

            if "metadata" in event:
                usage = _usage_from_wire(event["metadata"].get("usage"))
                if usage is not None:
                    yield _RawUsage(usage=usage)
        provider_state = _provider_state(reasoning_content)
        if provider_state is not None:
            yield _RawProviderState(provider_state=provider_state)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def _pump_chunks(
    iterator: Iterator[_RawChunk],
    loop,
    queue: Queue,
    stopped: Event,
) -> None:
    try:
        for chunk in iterator:
            if stopped.is_set():
                return
            loop.call_soon_threadsafe(queue.put_nowait, chunk)
    except Exception as error:  # noqa: BLE001 - transport errors cross the thread boundary
        if not stopped.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, error)
    finally:
        if not stopped.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)


class BedrockModel(BaseModel):
    """Toki backend for Amazon Bedrock Runtime's Converse APIs."""

    def __init__(
        self,
        model: BedrockModelName | str,
        api_key: str | None = None,
        *,
        profile_name: str | None = None,
        region_name: str | None = None,
        allow_parallel_tool_calls: bool = False,
        reasoning_effort: ReasoningEffort | None = None,
        reasoning_config: BedrockReasoningConfig | None = None,
        cache: Literal["rolling", "static"] | None = None,
        cache_ttl: Literal["5m", "1h"] = "5m",
    ):
        super().__init__()
        if reasoning_effort is not None and reasoning_config is not None:
            raise ValueError("reasoning_effort and reasoning_config cannot both be set")
        if api_key is not None:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        self.model = model
        self.api_key = api_key
        self.profile_name = profile_name
        self.region_name = region_name
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.reasoning_effort = reasoning_effort
        self.reasoning_config = reasoning_config
        self.cache = cache
        self.cache_ttl = cache_ttl
        attributes = get_bedrock_model_attributes(model)
        self._cache_state = _CacheState(
            min_cache_size_estimate=(
                attributes.cache_min_tokens
                if attributes is not None and attributes.cache_min_tokens is not None
                else 1_024
            )
        )
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self._session = session
        self.region_name = getattr(session, "region_name", region_name)
        client_kwargs: dict = {}
        if os.getenv("AWS_BEARER_TOKEN_BEDROCK") is not None:
            client_kwargs["config"] = Config(auth_scheme_preference="httpBearerAuth")
        self._client = session.client("bedrock-runtime", **client_kwargs)

    def _get_allow_parallel_tool_calls(self) -> bool:
        return self.allow_parallel_tool_calls

    def _attributes_map(self) -> dict:
        attributes = get_bedrock_model_attributes(self.model)
        if attributes is not None and self.model not in attributes_map:
            return {**attributes_map, self.model: attributes}
        return attributes_map

    def _supports_thinking(self) -> bool | None:
        attributes = get_bedrock_model_attributes(self.model)
        return attributes.supports_thinking if attributes is not None else None

    def invalidate_cache(self) -> None:
        """Forget every static and rolling Bedrock cache anchor."""
        self._cache_state.clear()

    def _apply_caching(
        self,
        *,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        system: list[dict] | None,
        wire_messages: list[dict],
        tool_config: dict | None,
    ) -> tuple[list[dict] | None, list[dict], dict | None]:
        if self.cache is None:
            return system, wire_messages, tool_config
        attributes = get_bedrock_model_attributes(self.model)
        if attributes is None:
            self._maybe_warn(
                "bedrock_cache_unknown",
                f"Bedrock explicit caching is unknown for {self.model!r}; "
                "sending the request without a cachePoint.",
                category=TokiCacheWarning,
                stacklevel=5,
            )
            return system, wire_messages, tool_config
        if not attributes.supports_explicit_caching:
            self._maybe_warn(
                "bedrock_cache_unsupported",
                f"Bedrock Converse explicit caching is not supported for "
                f"{self.model!r}; implicit provider caching may still apply.",
                category=TokiCacheWarning,
                stacklevel=5,
            )
            return system, wire_messages, tool_config
        if self.cache_ttl not in attributes.cache_ttls:
            self._maybe_warn(
                "bedrock_cache_ttl_unsupported",
                f"Bedrock model {self.model!r} does not support cache_ttl="
                f"{self.cache_ttl!r}; sending the request without a cachePoint.",
                category=TokiCacheWarning,
                stacklevel=5,
            )
            return system, wire_messages, tool_config

        candidate_anchor = (
            len(messages) if self.cache == "static" else len(messages) - 1
        )
        entry = self._cache_state.match_or_snapshot(
            strategy=self.cache,
            messages=messages,
            system=None,
            tools=tools,
            prefix_token_estimate=estimate_messages_tokens(
                None,
                tools,
                messages[:candidate_anchor],
            ),
            refresh_delta_tokens=1 if self.cache == "rolling" else 0,
        )
        if entry is None:
            return system, wire_messages, tool_config
        _, prefix_messages = _messages_to_wire(messages[: entry.anchor_index])
        return apply_cache_point(
            system=system,
            messages=wire_messages,
            tool_config=tool_config,
            boundary_index=len(prefix_messages) - 1,
            ttl=self.cache_ttl,
            locations=attributes.cache_locations,
        )

    def _reasoning_kwargs(
        self,
        kwargs: dict,
        *,
        capture_thinking: bool,
    ) -> dict:
        out = dict(kwargs)
        additional = dict(out.get("additionalModelRequestFields", {}))
        native_reasoning_keys = {
            "thinking",
            "reasoning",
            "reasoningConfig",
            "reasoning_effort",
        }
        if native_reasoning_keys & additional.keys():
            return out

        config = self.reasoning_config
        attributes = get_bedrock_model_attributes(self.model)
        family = attributes.reasoning_family if attributes is not None else None
        if config is None:
            effort = self.reasoning_effort
            if effort is None and capture_thinking:
                effort = "medium"
            if effort is None:
                return out
            if family is None:
                self._maybe_warn(
                    "bedrock_reasoning_mapping_unknown",
                    f"Bedrock reasoning controls are unknown for {self.model!r}; "
                    "pass a typed reasoning_config or native "
                    "additionalModelRequestFields.",
                    category=TokiBackendQuirkWarning,
                    stacklevel=5,
                )
                return out
            config = default_reasoning_config(family, effort)
        elif family is not None and reasoning_family(config) != family:
            raise ValueError(
                f"{type(config).__name__} does not match the {family!r} "
                f"reasoning family for {self.model!r}"
            )

        additional = {
            **reasoning_request_fields(config),
            **additional,
        }
        out["additionalModelRequestFields"] = additional
        if isinstance(config, ClaudeBudgetReasoning):
            inference_config = dict(out.get("inferenceConfig", {}))
            inference_config.setdefault(
                "maxTokens",
                config.budget_tokens + 1_024,
            )
            out["inferenceConfig"] = inference_config
        return out

    def _request(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        kwargs: dict,
        *,
        capture_thinking: bool = False,
        apply_caching: bool = True,
    ) -> dict:
        kwargs = self._reasoning_kwargs(
            kwargs,
            capture_thinking=capture_thinking,
        )
        system, wire_messages = _messages_to_wire(messages)
        native_tool_config = kwargs.pop("toolConfig", None)
        tool_config = _tools_to_wire(tools)
        if native_tool_config is not None:
            if tool_config is None:
                tool_config = dict(native_tool_config)
            else:
                tool_config = {
                    **native_tool_config,
                    "tools": tool_config["tools"],
                }
        if apply_caching:
            system, wire_messages, tool_config = self._apply_caching(
                messages=messages,
                tools=tools,
                system=system,
                wire_messages=wire_messages,
                tool_config=tool_config,
            )
        request: dict = {
            "modelId": self.model,
            "messages": wire_messages,
        }
        if system is not None:
            request["system"] = system
        if tool_config is not None:
            request["toolConfig"] = tool_config
        request.update(kwargs)
        return request

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal["exact", "online"] = "exact",
    ) -> int:
        if kind not in {"exact", "online"}:
            raise ValueError(
                f"BedrockModel supports kind='exact' and kind='online'; got {kind!r}"
            )
        normalized = [TokiMessage.from_dict(message) for message in messages]
        wire_tools, _ = _unwrap_tools(tools)
        request = self._request(
            normalized,
            wire_tools,
            {},
            apply_caching=False,
        )
        converse_input = {
            key: value
            for key, value in request.items()
            if key in {"messages", "system", "toolConfig"}
        }
        try:
            response = self._client.count_tokens(
                modelId=self.model,
                input={"converse": converse_input},
            )
            return response["inputTokens"]
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            is_profile = (
                self.model.startswith(("apac.", "eu.", "global.", "us."))
                or ":inference-profile/" in self.model
            )
            if code != "ValidationException" or not is_profile:
                raise

        normalized_id = self.model.rsplit("/", 1)[-1]
        if ".anthropic." in f".{normalized_id}" and self.region_name is not None:
            return count_claude_tokens(
                session=self._session,
                region_name=self.region_name,
                model_id=self.model,
                api_key=self.api_key,
                messages=normalized,
                tools=wire_tools,
            )

        self._maybe_warn(
            "bedrock_count_tokens_inference_fallback",
            f"Bedrock CountTokens does not support profile {self.model!r}; "
            "using a one-output-token Converse request instead.",
            category=TokiBackendQuirkWarning,
            stacklevel=4,
        )
        fallback = self._client.converse(
            **self._request(
                normalized,
                wire_tools,
                {"inferenceConfig": {"maxTokens": 1}},
                apply_caching=False,
            )
        )
        usage = _usage_from_wire(fallback["usage"])
        assert usage is not None
        return usage.prompt_tokens

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal["exact", "online"] = "exact",
    ) -> int:
        return await asyncio.to_thread(
            self.count_tokens,
            messages,
            tools=tools,
            kind=kind,
        )

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        response = self._client.converse(
            **self._request(
                messages,
                tools,
                kwargs,
                capture_thinking=capture_thinking,
            )
        )
        return _turn_from_response(response, capture_thinking=capture_thinking)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        response = self._client.converse_stream(
            **self._request(
                messages,
                tools,
                kwargs,
                capture_thinking=capture_thinking,
            )
        )
        yield from _stream_chunks(response["stream"])

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        return await asyncio.to_thread(
            self._raw_blocking,
            messages,
            tools,
            capture_thinking=capture_thinking,
            **kwargs,
        )

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        response = await asyncio.to_thread(
            self._client.converse_stream,
            **self._request(
                messages,
                tools,
                kwargs,
                capture_thinking=capture_thinking,
            ),
        )
        chunks = _stream_chunks(response["stream"])
        queue: Queue = Queue()
        stopped = Event()
        Thread(
            target=_pump_chunks,
            args=(chunks, asyncio.get_running_loop(), queue, stopped),
            daemon=True,
        ).start()
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            stopped.set()
            response["stream"].close()
