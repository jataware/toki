import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Literal, overload


Role = Literal["user", "assistant", "system", "tool"]


@dataclass
class TokiToolFunction:
    name: str
    arguments: str  # JSON-encoded; convert via json.loads

    @classmethod
    def from_dict(cls, x: 'TokiToolFunction | dict') -> 'TokiToolFunction':
        return x if isinstance(x, cls) else cls(**x)


@dataclass
class TokiToolCall:
    id: str
    function: TokiToolFunction
    type: Literal["function"] = "function"

    @classmethod
    def from_dict(cls, x: 'TokiToolCall | dict') -> 'TokiToolCall':
        if isinstance(x, cls):
            return x
        return cls(
            id=x["id"],
            type=x.get("type", "function"),
            function=TokiToolFunction.from_dict(x["function"]),
        )


@dataclass
class TokiMessage:
    role: Role
    content: str
    tool_calls: list[TokiToolCall] | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, x: 'TokiMessage | dict') -> 'TokiMessage':
        if isinstance(x, cls):
            return x
        tcs = x.get("tool_calls")
        return cls(
            role=x["role"],
            content=x["content"],
            tool_calls=[TokiToolCall.from_dict(t) for t in tcs] if tcs else None,
            tool_call_id=x.get("tool_call_id"),
        )


@dataclass
class TokiUsageMetadata:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TokiChatResponse:
    """Blocking-mode chat response when `capture_thinking=True`."""
    content: str
    thought: str  # empty string if model produced no thinking


@dataclass
class TokiToolResponse:
    """Returned (blocking) or yielded (streaming) when the model emits tool calls."""
    thought: str
    tool_calls: list[TokiToolCall]


@dataclass
class TokiThinking:
    """Streaming chunk carrying a piece of the model's reasoning text. Only seen when `capture_thinking=True`."""
    text: str


def pretty_tool_call(tool_call: TokiToolCall) -> str:
    """Return a string representation of the tool call, i.e. `tool_name(arg1=value1, arg2=value2, ...)`"""
    args = json.loads(tool_call.function.arguments)
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    return f'{tool_call.function.name}({args_str})'


class BaseModel(ABC):
    """Abstract base for all toki model backends. Subclasses implement `_blocking_complete` and `_streaming_complete`."""

    def __init__(self) -> None:
        # updated after every completion
        self._usage_metadata: TokiUsageMetadata | None = None

    # blocking
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> str: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: list, capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolResponse: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: list, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse | TokiToolResponse: ...
    # streaming
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: list, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: list, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    def complete(
        self,
        messages: list[TokiMessage | dict],
        *,
        stream: bool = False,
        tools: list | None = None,
        capture_thinking: bool = False,
        **kwargs,
    ) -> str | TokiChatResponse | TokiToolResponse | Generator[str | TokiThinking | TokiToolResponse, None, None]:
        normalized = [TokiMessage.from_dict(m) for m in messages]
        if stream:
            return self._streaming_complete(normalized, tools, capture_thinking=capture_thinking, **kwargs)
        return self._blocking_complete(normalized, tools, capture_thinking=capture_thinking, **kwargs)

    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse | TokiToolResponse: ...
    @abstractmethod
    def _blocking_complete(self, messages: list[TokiMessage], tools: list | None = None, *, capture_thinking: bool = False, **kwargs) -> str | TokiChatResponse | TokiToolResponse: ...

    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    @abstractmethod
    def _streaming_complete(self, messages: list[TokiMessage], tools: list | None = None, *, capture_thinking: bool = False, **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
