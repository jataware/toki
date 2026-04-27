from typing import Generator, Generic, Literal, TypeVar, cast, overload

from .model import (
    BaseModel,
    Role,
    TokiChatResponse,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolResponse,
)


# set up type-hinting such that if the user doesn't provide tools, then it's just str, not str|TokiToolResponse
class WithTools: ...
class WithoutTools: ...
HasTools = TypeVar('HasTools', WithTools, WithoutTools)


# TODO: consider renaming to e.g. Chat or something similar, and reserve Agent for ReAct agents
class Agent(Generic[HasTools]):
    """Basically just a model paired with message history tracking"""
    @overload
    def __init__(self: 'Agent[WithoutTools]', model: BaseModel, tools: None = None): ...
    @overload
    def __init__(self: 'Agent[WithTools]', model: BaseModel, tools: list): ...
    def __init__(self, model: BaseModel, tools: list | None = None):
        self.model = model
        self.messages: list[TokiMessage] = []
        self.tools = tools

    @overload
    def add_message(self: 'Agent[WithTools]|Agent[WithoutTools]', *, role: Role, content: str): ...
    @overload
    def add_message(self: 'Agent[WithTools]', *, role: Role, content: str, tool_call_id: str): ...
    @overload
    def add_message(self: 'Agent[WithTools]', *, role: Role, content: str, tool_calls: list[TokiToolCall | dict]): ...
    def add_message(self, *, role: Role, content: str, tool_calls: list[TokiToolCall | dict] | None = None, tool_call_id: str | None = None):
        assert tool_calls is None or tool_call_id is None, "tool_calls and tool_call_id cannot both be provided"
        if tool_calls:
            message = TokiMessage(role=role, content=content, tool_calls=[TokiToolCall.from_dict(tc) for tc in tool_calls])
        elif tool_call_id:
            message = TokiMessage(role=role, content=content, tool_call_id=tool_call_id)
        else:
            message = TokiMessage(role=role, content=content)
        self.messages.append(message)

    def add_user_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='user', content=content)

    def add_assistant_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='assistant', content=content)

    def add_assistant_tool_calls(self: 'Agent[WithTools]', content: str, tool_calls: list[TokiToolCall | dict]):
        self.add_message(role='assistant', content=content, tool_calls=tool_calls)

    def add_tool_message(self: 'Agent[WithTools]', tool_call_id: str, content: str):
        self.add_message(role='tool', tool_call_id=tool_call_id, content=content)

    def add_system_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='system', content=content)

    # blocking
    @overload
    def execute(self: 'Agent[WithTools]', stream: Literal[False] = False, *, capture_thinking: Literal[False] = False) -> str | TokiToolResponse: ...
    @overload
    def execute(self: 'Agent[WithoutTools]', stream: Literal[False] = False, *, capture_thinking: Literal[False] = False) -> str: ...
    @overload
    def execute(self: 'Agent[WithTools]', stream: Literal[False] = False, *, capture_thinking: Literal[True]) -> TokiChatResponse | TokiToolResponse: ...
    @overload
    def execute(self: 'Agent[WithoutTools]', stream: Literal[False] = False, *, capture_thinking: Literal[True]) -> TokiChatResponse: ...
    # streaming
    @overload
    def execute(self: 'Agent[WithTools]', stream: Literal[True], *, capture_thinking: Literal[False] = False) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def execute(self: 'Agent[WithoutTools]', stream: Literal[True], *, capture_thinking: Literal[False] = False) -> Generator[str, None, None]: ...
    @overload
    def execute(self: 'Agent[WithTools]', stream: Literal[True], *, capture_thinking: Literal[True]) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    @overload
    def execute(self: 'Agent[WithoutTools]', stream: Literal[True], *, capture_thinking: Literal[True]) -> Generator[str | TokiThinking, None, None]: ...
    def execute(
        self: 'Agent[WithTools]|Agent[WithoutTools]',
        stream: bool = False,
        *,
        capture_thinking: bool = False,
    ) -> str | TokiChatResponse | TokiToolResponse | Generator[str | TokiThinking | TokiToolResponse, None, None]:
        if stream:
            return self._streaming_execute(capture_thinking=capture_thinking)
        return self._blocking_execute(capture_thinking=capture_thinking)

    @overload
    def _blocking_execute(self: 'Agent[WithTools]', *, capture_thinking: Literal[False] = False) -> str | TokiToolResponse: ...
    @overload
    def _blocking_execute(self: 'Agent[WithoutTools]', *, capture_thinking: Literal[False] = False) -> str: ...
    @overload
    def _blocking_execute(self: 'Agent[WithTools]', *, capture_thinking: Literal[True]) -> TokiChatResponse | TokiToolResponse: ...
    @overload
    def _blocking_execute(self: 'Agent[WithoutTools]', *, capture_thinking: Literal[True]) -> TokiChatResponse: ...
    def _blocking_execute(self: 'Agent[WithTools]|Agent[WithoutTools]', *, capture_thinking: bool = False) -> str | TokiChatResponse | TokiToolResponse:
        # if here is mainly for type hinting since apparently it doesn't know how to merge the cases where self.tools is None|list
        if self.tools is None:
            result = self.model.complete(self.messages, stream=False, capture_thinking=capture_thinking)
        else:
            result = self.model.complete(self.messages, stream=False, tools=self.tools, capture_thinking=capture_thinking)
        if isinstance(result, str):
            self.add_assistant_message(result)
        elif isinstance(result, TokiChatResponse):
            self.add_assistant_message(result.content)
        else:
            self = cast(Agent[WithTools], self)  # TODO: is there a better way to narrow this
            self.add_assistant_tool_calls(result.thought, result.tool_calls)
        return result

    @overload
    def _streaming_execute(self: 'Agent[WithTools]', *, capture_thinking: Literal[False] = False) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def _streaming_execute(self: 'Agent[WithoutTools]', *, capture_thinking: Literal[False] = False) -> Generator[str, None, None]: ...
    @overload
    def _streaming_execute(self: 'Agent[WithTools]', *, capture_thinking: Literal[True]) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    @overload
    def _streaming_execute(self: 'Agent[WithoutTools]', *, capture_thinking: Literal[True]) -> Generator[str | TokiThinking, None, None]: ...
    def _streaming_execute(self: 'Agent[WithTools]|Agent[WithoutTools]', *, capture_thinking: bool = False) -> Generator[str | TokiThinking | TokiToolResponse, None, None]:
        # stream the chunks while also capturing them
        result_chunks: list[str] = []
        tool_responses: list[TokiToolResponse] = []
        for chunk in self.model.complete(self.messages, stream=True, tools=self.tools, capture_thinking=capture_thinking):
            if isinstance(chunk, TokiToolResponse):
                tool_responses.append(chunk)
            elif isinstance(chunk, TokiThinking):
                pass  # ephemeral; not added to message history
            else:
                result_chunks.append(chunk)
            yield chunk

        # add the message to the history after streaming is done
        if tool_responses:
            self = cast(Agent[WithTools], self)  # TODO: is there a better way to narrow this
            for tool_response in tool_responses:
                self.add_assistant_tool_calls(tool_response.thought, tool_response.tool_calls)
            if result_chunks:
                self.add_assistant_message(''.join(result_chunks))
        else:
            self.add_assistant_message(''.join(result_chunks))
