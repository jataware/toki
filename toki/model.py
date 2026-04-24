import json
from abc import ABC, abstractmethod
from typing import Generator, Literal, TypedDict, overload

from typing_extensions import NotRequired


Role = Literal["user", "assistant", "system", "tool"]


class TokiToolFunction(TypedDict):
    name: str
    arguments: str  # needs to be converted to dict via json.loads


class TokiToolCall(TypedDict):
    id: str
    type: Literal["function"]  # TODO: other types?
    function: TokiToolFunction


class TokiMessage(TypedDict):
    role: Role
    content: str
    tool_calls: NotRequired[list[TokiToolCall]]
    tool_call_id: NotRequired[str]


class TokiToolResponse(TypedDict):
    thought: str
    tool_calls: list[TokiToolCall]


# TODO: more usage metadata can be added as backends report it
class TokiUsageMetadata(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def pretty_tool_call(tool_call: TokiToolCall) -> str:
    """Return a string representation of the tool call, i.e. `tool_name(arg1=value1, arg2=value2, ...)`"""
    args = json.loads(tool_call["function"]["arguments"])
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    return f'{tool_call["function"]["name"]}({args_str})'


class BaseModel(ABC):
    """Abstract base for all toki model backends. Subclasses implement `_blocking_complete` and `_streaming_complete`."""

    def __init__(self) -> None:
        # updated after every completion
        self._usage_metadata: TokiUsageMetadata | None = None

    @overload
    def complete(self, messages: list[TokiMessage], *, stream: Literal[False] = False, tools: None = None, **kwargs) -> str: ...
    @overload
    def complete(self, messages: list[TokiMessage], *, stream: Literal[False] = False, tools: list, **kwargs) -> str | TokiToolResponse: ...
    @overload
    def complete(self, messages: list[TokiMessage], *, stream: Literal[True], tools: None = None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage], *, stream: Literal[True], tools: list, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    def complete(self, messages: list[TokiMessage], *, stream: bool = False, tools: list | None = None, **kwargs) -> str | TokiToolResponse | Generator[str | TokiToolResponse, None, None]:
        if stream:
            return self._streaming_complete(messages, tools, **kwargs)
        return self._blocking_complete(messages, tools, **kwargs)

    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> str | TokiToolResponse: ...
    @abstractmethod
    def _blocking_complete(self, messages: list[TokiMessage], tools: list | None = None, **kwargs) -> str | TokiToolResponse: ...

    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    @abstractmethod
    def _streaming_complete(self, messages: list[TokiMessage], tools: list | None = None, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
