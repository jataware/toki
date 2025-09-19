import requests
import json
from typing import Generator, Literal, overload, TypedDict, Any, Optional, Sequence, cast, Generic, TypeVar
from typing_extensions import NotRequired
from pathlib import Path
from functools import lru_cache
import logging

here = Path(__file__).parent
logger = logging.getLogger(__name__)


from .openrouter_models import ModelName, attributes_map

import pdb

Role = Literal["user", "assistant", 'system', 'tool']

class OpenRouterMessage(TypedDict):
    role: Role
    content: str
    tool_calls: 'NotRequired[list[OpenRouterToolCall]]'
    tool_call_id: NotRequired[str]

class OpenRouterToolResponse(TypedDict):
    thought: str
    tool_calls: 'list[OpenRouterToolCall]'

class OpenRouterToolCall(TypedDict):
    id: str
    type: Literal["function"]  # TODO: other types?
    function: 'OpenRouterToolFunction'

class OpenRouterToolFunction(TypedDict):
    name: str
    arguments: str # needs to be converted to dict via json.loads

class OpenRouterResponse(TypedDict):
    choices: 'list[OpenRouterResponseCompletionChoice]'
    usage: 'OpenRouterUsageMetadata'

class OpenRouterResponseCompletionChoice(TypedDict):
    message: OpenRouterMessage

class OpenRouterResponseDeltaPayload(TypedDict):
    content: NotRequired[str]
    tool_calls: NotRequired[list[OpenRouterToolCall]]

class OpenRouterResponseChoice(TypedDict):
    delta: OpenRouterResponseDeltaPayload

class OpenRouterResponseDelta(TypedDict):
    choices: list[OpenRouterResponseChoice]
    usage: NotRequired['OpenRouterUsageMetadata']   # typically only on the final chunk

class OpenRouterResponseError(TypedDict):
    error: Any


# TODO: there is more usage metadata not captured here, can expand if we want access to it
#       see: https://openrouter.ai/docs/use-cases/usage-accounting#response-format
class OpenRouterUsageMetadata(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def pretty_tool_call(tool_call: OpenRouterToolCall) -> str:
    """Return a string representation of the tool call, i.e. `tool_name(arg1=value1, arg2=value2, ...)`"""
    args = json.loads(tool_call["function"]["arguments"])
    args_str = ', '.join([f'{k}={v}' for k, v in args.items()])
    return f'{tool_call["function"]["name"]}({args_str})'

class Model:
    """A simple class for talking to OpenRouter models directly via requests to the API"""
    def __init__(self, model:ModelName, openrouter_api_key:str, allow_parallel_tool_calls:bool=False):
        self.model = model
        self.openrouter_api_key = openrouter_api_key
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

        # updated after every completion
        self._usage_metadata: OpenRouterUsageMetadata|None = None


    @overload
    def complete(self, messages: list[OpenRouterMessage], *, stream:Literal[False]=False, tools:None=None, **kwargs) -> str: ...
    @overload
    def complete(self, messages: list[OpenRouterMessage], *, stream:Literal[False]=False, tools:list, **kwargs) -> str | OpenRouterToolResponse: ...
    @overload
    def complete(self, messages: list[OpenRouterMessage], *, stream:Literal[True], tools:None=None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def complete(self, messages: list[OpenRouterMessage], *, stream:Literal[True], tools:list, **kwargs) -> Generator[str | OpenRouterToolResponse, None, None]: ...
    def complete(self, messages: list[OpenRouterMessage], *, stream:bool=False, tools:list|None=None, **kwargs) -> str | OpenRouterToolResponse | Generator[str | OpenRouterToolResponse, None, None]:
        if stream:
            return self._streaming_complete(messages, tools, **kwargs)
        else:
            return self._blocking_complete(messages, tools, **kwargs)


    @overload
    def _blocking_complete(self, messages: list[OpenRouterMessage], tools:None=None, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[OpenRouterMessage], tools:list, **kwargs) -> str | OpenRouterToolResponse: ...
    def _blocking_complete(self, messages: list[OpenRouterMessage], tools:list|None=None, **kwargs) -> str | OpenRouterToolResponse:
        tool_payload = {"tools": tools, "parallel_tool_calls": self.allow_parallel_tool_calls, 'tool_choice': {"type": "function", "function": {"name": "test_tool"}}} if tools else {}
        payload = {"model": self.model, "messages": messages, **tool_payload, **kwargs}
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            data=json.dumps(payload)
        )
        data = cast(OpenRouterResponse|OpenRouterResponseError, response.json())
        if 'error' in data:
            pdb.set_trace()
            raise ValueError(f"Error from OpenRouter: {data}")
        try:
            self._usage_metadata = cast(OpenRouterUsageMetadata, data['usage'])
            if 'tool_calls' in data['choices'][0]['message'] and len(data['choices'][0]['message']['tool_calls']) > 0:
                return OpenRouterToolResponse(thought=data['choices'][0]['message']['content'], tool_calls=data['choices'][0]['message']['tool_calls'])
            return data['choices'][0]['message']['content']
        except KeyError as e:
            raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
        except Exception as e:
            raise ValueError(f"An error occurred while processing the response: '{data}'. {e}") from e

    # TODO: should request timeout be a setting rather than hardcoded?
    @overload
    def _streaming_complete(self, messages: list[OpenRouterMessage], tools:None=None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[OpenRouterMessage], tools:list, **kwargs) -> Generator[str|OpenRouterToolResponse, None, None]: ...
    def _streaming_complete(self, messages: list[OpenRouterMessage], tools:list|None=None, **kwargs) -> Generator[str|OpenRouterToolResponse, None, None]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        tool_payload = {"tools": tools, "parallel_tool_calls": False} if tools else {}
        payload = {"model": self.model, "messages": messages, "stream": True, **tool_payload, **kwargs}

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            buf = []
            for line in r.iter_lines(decode_unicode=True, chunk_size=1024):
                line = cast(str|None, line)
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
                        obj = cast(OpenRouterResponseDelta|OpenRouterResponseError, json.loads(data))
                    except json.JSONDecodeError:
                        continue # wait for the next complete event
                    if 'error' in obj:
                        raise ValueError(f"Error from OpenRouter: {obj}")
                    try:
                        content: str|None = obj["choices"][0]["delta"].get("content")
                        tool_calls: list[OpenRouterToolCall]|None = obj["choices"][0]["delta"].get("tool_calls")
                        if tool_calls:
                            yield OpenRouterToolResponse(thought=content or '', tool_calls=tool_calls)
                        elif content:
                            yield content
                    except KeyError as e:
                        raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
                    
                    # update the usage metadata (typically on the final chunk)
                    if "usage" in obj:
                        self._usage_metadata = cast(OpenRouterUsageMetadata, obj["usage"])  # type: ignore[index]
                    
                    continue

                
                # ignore other fields like "event:" / "id:" / comments / etc.


# TODO: make wrapper class around Model that interfaces with tools, but as strings rather than via the openrouter API
#       basically for cases where the model either doesn't support tools, or it does but the interface is flaky
#       it should be usable as a drop-in replacement for Model (e.g. in Agent/etc.)






# from .model import Model, Message, Role


# set up type-hinting such that if the user doesn't provide tools, then it's just str, not str|OpenRouterToolResponse
class WithTools: ...
class WithoutTools: ...
HasTools = TypeVar('HasTools', WithTools, WithoutTools)
# TODO: consider renaming to e.g. Chat or something similar, and reserve Agent for ReAct agents
class Agent(Generic[HasTools]):
    """Basically just a model paired with message history tracking"""
    @overload
    def __init__(self: 'Agent[WithoutTools]', model: Model, tools:None=None): ...
    @overload
    def __init__(self: 'Agent[WithTools]', model: Model, tools:list): ...
    def __init__(self, model: Model, tools:list|None=None):
        self.model = model
        self.messages: list[OpenRouterMessage] = []
        self.tools = tools

    @overload
    def add_message(self: 'Agent[WithTools]|Agent[WithoutTools]', *, role: Role, content: str): ...
    @overload
    def add_message(self: 'Agent[WithTools]', *, role: Role, content: str, tool_call_id: str): ...
    @overload
    def add_message(self: 'Agent[WithTools]', *, role: Role, content: str, tool_calls: list[OpenRouterToolCall]): ...
    def add_message(self, *, role: Role, content: str, tool_calls: list[OpenRouterToolCall]|None=None, tool_call_id: str|None=None):
        assert tool_calls is None or tool_call_id is None, "tool_calls and tool_call_id cannot both be provided"
        if tool_calls:
            message = OpenRouterMessage(role=role, content=content, tool_calls=tool_calls)
        elif tool_call_id:
            message = OpenRouterMessage(role=role, content=content, tool_call_id=tool_call_id)
        else:
            message = OpenRouterMessage(role=role, content=content)
        self.messages.append(message)
    
    def add_user_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='user', content=content)

    def add_assistant_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='assistant', content=content)
    
    def add_assistant_tool_calls(self: 'Agent[WithTools]', content: str, tool_calls: list[OpenRouterToolCall]):
        self.add_message(role='assistant', content=content, tool_calls=tool_calls)

    def add_tool_message(self: 'Agent[WithTools]', tool_call_id: str, content: str):
        self.add_message(role='tool', tool_call_id=tool_call_id, content=content)

    def add_system_message(self: 'Agent[WithTools]|Agent[WithoutTools]', content: str):
        self.add_message(role='system', content=content)
    
    @overload
    def execute(self:'Agent[WithTools]', stream:Literal[False]=False) -> str|OpenRouterToolResponse: ...
    @overload
    def execute(self:'Agent[WithTools]', stream:Literal[True]) -> Generator[str | OpenRouterToolResponse, None, None]: ...
    @overload
    def execute(self:'Agent[WithoutTools]', stream:Literal[False]=False) -> str: ...
    @overload
    def execute(self:'Agent[WithoutTools]', stream:Literal[True]) -> Generator[str, None, None]: ...
    def execute(self:'Agent[WithTools]|Agent[WithoutTools]', stream:bool=False) -> str | OpenRouterToolResponse | Generator[str | OpenRouterToolResponse, None, None]:
        if stream:
            return self._streaming_execute()
        else:
            return self._blocking_execute()
    
    @overload
    def _blocking_execute(self:'Agent[WithTools]') -> str | OpenRouterToolResponse: ...
    @overload
    def _blocking_execute(self:'Agent[WithoutTools]') -> str: ...
    def _blocking_execute(self:'Agent[WithTools]|Agent[WithoutTools]') -> str | OpenRouterToolResponse:
        # if here is mainly for type hinting since apparently it doesn't know how to merge the cases where self.tools is None|list
        if self.tools is None:
            result = self.model.complete(self.messages, stream=False)
        else:
            result = self.model.complete(self.messages, stream=False, tools=self.tools)
        if isinstance(result, str):
            self.add_assistant_message(result)
        else:
            self = cast(Agent[WithTools], self) #TODO: is there a better way to narrow this
            self.add_assistant_tool_calls(result['thought'], result['tool_calls'])
        return result

    @overload
    def _streaming_execute(self:'Agent[WithTools]') -> Generator[str|OpenRouterToolResponse, None, None]: ...
    @overload
    def _streaming_execute(self:'Agent[WithoutTools]') -> Generator[str, None, None]: ...
    def _streaming_execute(self:'Agent[WithTools]|Agent[WithoutTools]') -> Generator[str|OpenRouterToolResponse, None, None]:
        # stream the chunks while also capturing them
        result_chunks = []
        tool_calls: list[OpenRouterToolResponse] = []
        for chunk in self.model.complete(self.messages, stream=True, tools=self.tools):
            if isinstance(chunk, dict):
                tool_calls.append(chunk)
            else:
                result_chunks.append(chunk)
            yield chunk
        
        # add the message to the history after streaming is done
        if tool_calls:
            self = cast(Agent[WithTools], self) #TODO: is there a better way to narrow this
            for tool_call in tool_calls:
                self.add_assistant_tool_calls(tool_call['thought'], tool_call['tool_calls'])
            if result_chunks:
                self.add_assistant_message(''.join(result_chunks))
        else:
            self.add_assistant_message(''.join(result_chunks))