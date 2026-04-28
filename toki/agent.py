from typing import Generator, Generic, Literal, Sequence, TypeVar, cast, overload

from .model import (
    BaseModel,
    Role,
    StreamingToolSchema,
    TokiMessage,
    TokiThinking,
    TokiThoughtResponse,
    TokiToolCall,
    TokiToolCallStream,
    TokiToolFunction,
    TokiToolsResponse,
    TokiToolsThoughtResponse,
    ToolSchema,
)


# Tools-shape markers for static-typing the Agent's capabilities.
class WithoutTools: ...
class WithStaticTools: ...
class WithStreamingTools: ...
class WithMixedTools: ...

ToolsShape = TypeVar('ToolsShape', WithoutTools, WithStaticTools, WithStreamingTools, WithMixedTools)
WithTools = WithStaticTools | WithStreamingTools | WithMixedTools  # convenience union


# TODO: consider renaming to e.g. Chat or something similar, and reserve Agent for ReAct agents
class Agent(Generic[ToolsShape]):
    """A model paired with message-history tracking. The tools-shape type parameter
    tracks whether the agent has no tools (`WithoutTools`), static-only tools
    (`WithStaticTools`), streaming-only tools (`WithStreamingTools`), or a mix
    (`WithMixedTools`); `execute()`'s return types specialize accordingly.
    """

    @overload
    def __init__(self: 'Agent[WithoutTools]', model: BaseModel, tools: None = None): ...
    @overload
    def __init__(self: 'Agent[WithStreamingTools]', model: BaseModel, tools: Sequence[StreamingToolSchema]): ...
    @overload
    def __init__(self: 'Agent[WithStaticTools]', model: BaseModel, tools: Sequence[ToolSchema | dict]): ...
    @overload
    def __init__(self: 'Agent[WithMixedTools]', model: BaseModel, tools: Sequence[StreamingToolSchema | ToolSchema | dict]): ...
    def __init__(self, model: BaseModel, tools: Sequence | None = None):
        self.model = model
        self.messages: list[TokiMessage] = []
        self.tools = tools

    @overload
    def add_message(self, *, role: Role, content: str): ...
    @overload
    def add_message(self, *, role: Role, content: str, tool_call_id: str): ...
    @overload
    def add_message(self, *, role: Role, content: str, tool_calls: list[TokiToolCall | dict]): ...
    def add_message(self, *, role: Role, content: str, tool_calls: list[TokiToolCall | dict] | None = None, tool_call_id: str | None = None):
        assert tool_calls is None or tool_call_id is None, "tool_calls and tool_call_id cannot both be provided"
        if tool_calls:
            message = TokiMessage(role=role, content=content, tool_calls=[TokiToolCall.from_dict(tc) for tc in tool_calls])
        elif tool_call_id:
            message = TokiMessage(role=role, content=content, tool_call_id=tool_call_id)
        else:
            message = TokiMessage(role=role, content=content)
        self.messages.append(message)

    def add_user_message(self, content: str):
        self.add_message(role='user', content=content)

    def add_assistant_message(self, content: str):
        self.add_message(role='assistant', content=content)

    def add_assistant_tool_calls(self: 'Agent[WithStaticTools] | Agent[WithStreamingTools] | Agent[WithMixedTools]', content: str, tool_calls: list[TokiToolCall | dict]):
        self.add_message(role='assistant', content=content, tool_calls=tool_calls)

    def add_tool_message(self: 'Agent[WithStaticTools] | Agent[WithStreamingTools] | Agent[WithMixedTools]', tool_call_id: str, content: str):
        self.add_message(role='tool', tool_call_id=tool_call_id, content=content)

    def add_system_message(self, content: str):
        self.add_message(role='system', content=content)

    # ----- execute: 16 overloads ------------------------------------------------

    # blocking, capture_thinking=False
    @overload
    def execute(self: 'Agent[WithoutTools]', *, stream: Literal[False] = False, capture_thinking: Literal[False] = False) -> str: ...
    @overload
    def execute(self: 'Agent[WithStaticTools]', *, stream: Literal[False] = False, capture_thinking: Literal[False] = False) -> str | TokiToolsResponse[TokiToolCall]: ...
    @overload
    def execute(self: 'Agent[WithStreamingTools]', *, stream: Literal[False] = False, capture_thinking: Literal[False] = False) -> str | TokiToolsResponse[TokiToolCallStream]: ...
    @overload
    def execute(self: 'Agent[WithMixedTools]', *, stream: Literal[False] = False, capture_thinking: Literal[False] = False) -> str | TokiToolsResponse[TokiToolCall | TokiToolCallStream]: ...
    # blocking, capture_thinking=True
    @overload
    def execute(self: 'Agent[WithoutTools]', *, stream: Literal[False] = False, capture_thinking: Literal[True]) -> TokiThoughtResponse: ...
    @overload
    def execute(self: 'Agent[WithStaticTools]', *, stream: Literal[False] = False, capture_thinking: Literal[True]) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall]: ...
    @overload
    def execute(self: 'Agent[WithStreamingTools]', *, stream: Literal[False] = False, capture_thinking: Literal[True]) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCallStream]: ...
    @overload
    def execute(self: 'Agent[WithMixedTools]', *, stream: Literal[False] = False, capture_thinking: Literal[True]) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall | TokiToolCallStream]: ...
    # streaming, capture_thinking=False
    @overload
    def execute(self: 'Agent[WithoutTools]', *, stream: Literal[True], capture_thinking: Literal[False] = False) -> Generator[str, None, None]: ...
    @overload
    def execute(self: 'Agent[WithStaticTools]', *, stream: Literal[True], capture_thinking: Literal[False] = False) -> Generator[str | TokiToolCall, None, None]: ...
    @overload
    def execute(self: 'Agent[WithStreamingTools]', *, stream: Literal[True], capture_thinking: Literal[False] = False) -> Generator[str | TokiToolCallStream, None, None]: ...
    @overload
    def execute(self: 'Agent[WithMixedTools]', *, stream: Literal[True], capture_thinking: Literal[False] = False) -> Generator[str | TokiToolCall | TokiToolCallStream, None, None]: ...
    # streaming, capture_thinking=True
    @overload
    def execute(self: 'Agent[WithoutTools]', *, stream: Literal[True], capture_thinking: Literal[True]) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def execute(self: 'Agent[WithStaticTools]', *, stream: Literal[True], capture_thinking: Literal[True]) -> Generator[str | TokiThinking | TokiToolCall, None, None]: ...
    @overload
    def execute(self: 'Agent[WithStreamingTools]', *, stream: Literal[True], capture_thinking: Literal[True]) -> Generator[str | TokiThinking | TokiToolCallStream, None, None]: ...
    @overload
    def execute(self: 'Agent[WithMixedTools]', *, stream: Literal[True], capture_thinking: Literal[True]) -> Generator[str | TokiThinking | TokiToolCall | TokiToolCallStream, None, None]: ...

    def execute(self, *, stream: bool = False, capture_thinking: bool = False):
        if stream:
            return self._streaming_execute(capture_thinking=capture_thinking)
        return self._blocking_execute(capture_thinking=capture_thinking)

    def _blocking_execute(self, *, capture_thinking: bool):
        if self.tools is None:
            result = self.model.complete(self.messages, stream=False, capture_thinking=capture_thinking)
        else:
            result = self.model.complete(self.messages, stream=False, tools=self.tools, capture_thinking=capture_thinking)

        if isinstance(result, str):
            self.add_assistant_message(result)
            return result
        if isinstance(result, TokiThoughtResponse):
            self.add_assistant_message(result.content)
            return result
        # at this point: TokiToolsResponse / TokiToolsThoughtResponse
        materialized = [_materialize_tool_call(tc) for tc in result.tool_calls]
        self_with_tools = cast('Agent[WithStaticTools]', self)
        self_with_tools.add_assistant_tool_calls(result.content, materialized)
        return result

    def _streaming_execute(self, *, capture_thinking: bool):
        content_chunks: list[str] = []
        tool_calls: list[TokiToolCall] = []
        streams: list[TokiToolCallStream] = []

        if self.tools is None:
            source = self.model.complete(self.messages, stream=True, capture_thinking=capture_thinking)
        else:
            source = self.model.complete(self.messages, stream=True, tools=self.tools, capture_thinking=capture_thinking)

        for chunk in source:
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            elif isinstance(chunk, TokiToolCall):
                tool_calls.append(chunk)
            elif isinstance(chunk, TokiToolCallStream):
                streams.append(chunk)
            # TokiThinking is ephemeral; not added to history
            yield chunk

        # any unconsumed streaming tool calls — drain them now to materialize args for history
        for s in streams:
            tool_calls.append(_materialize_tool_call(s))

        content = ''.join(content_chunks)
        if tool_calls:
            self_with_tools = cast('Agent[WithStaticTools]', self)
            self_with_tools.add_assistant_tool_calls(content, tool_calls)
        else:
            self.add_assistant_message(content)


def _materialize_tool_call(tc: TokiToolCall | TokiToolCallStream) -> TokiToolCall:
    """Convert a tool-call entry from a response into a concrete `TokiToolCall`. For
    a `TokiToolCallStream` this drains the stream (idempotent) and reads the parsed
    arguments dict."""
    if isinstance(tc, TokiToolCall):
        return tc
    return TokiToolCall(id=tc.id, function=TokiToolFunction(name=tc.name, arguments=tc.arguments))
