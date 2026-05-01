"""Internal async-machinery tests that don't require any backend / API key.

Drives `BaseModel` via a fake backend that yields canned `_RawChunk`s, exercising
`_AsyncStreamDriver` + `AsyncTokiToolCallStream` + `AsyncTokiArgStream` end-to-end.
"""

import asyncio
import json
from typing import AsyncIterator, Iterator

import pytest

from toki import (
    AsyncTokiArgStream,
    AsyncTokiToolCallStream,
    BaseModel,
    StreamingToolSchema,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolsResponse,
    ToolSchema,
)
from toki.model import (
    _RawChunk,
    _RawContentChunk,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
)


class _FakeAsyncModel(BaseModel):
    """A model whose raw I/O methods replay a fixed sequence of `_RawChunk`s for streaming
    and a fixed `_RawTurn` for blocking. Lets us drive the async surface without a real backend.
    """

    def __init__(self, *, blocking_turn: _RawTurn | None = None, streaming_chunks: list[_RawChunk] | None = None):
        super().__init__()
        self._blocking_turn = blocking_turn
        self._streaming_chunks = streaming_chunks or []

    def _raw_blocking(self, messages, tools, *, capture_thinking, **kwargs) -> _RawTurn:
        assert self._blocking_turn is not None
        return self._blocking_turn

    def _raw_streaming(self, messages, tools, *, capture_thinking, **kwargs) -> Iterator[_RawChunk]:
        for c in self._streaming_chunks:
            yield c

    async def _raw_blocking_async(self, messages, tools, *, capture_thinking, **kwargs) -> _RawTurn:
        assert self._blocking_turn is not None
        return self._blocking_turn

    async def _raw_streaming_async(self, messages, tools, *, capture_thinking, **kwargs) -> AsyncIterator[_RawChunk]:
        for c in self._streaming_chunks:
            await asyncio.sleep(0)  # yield to the loop between chunks
            yield c


def _msg() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


# ----- blocking ----------------------------------------------------------


async def test_acomplete_blocking_plain_string():
    model = _FakeAsyncModel(blocking_turn=_RawTurn(content="hello", tool_calls=[], thought=""))
    result = await model.acomplete(_msg())
    assert result == "hello"


async def test_acomplete_blocking_with_static_tool_call():
    tc = TokiToolCall.from_dict({"id": "x", "function": {"name": "record", "arguments": json.dumps({"v": 1})}})
    model = _FakeAsyncModel(blocking_turn=_RawTurn(content="", tool_calls=[tc], thought=""))
    result = await model.acomplete(_msg(), tools=[ToolSchema(schema={"type": "function", "function": {"name": "record"}})])
    assert isinstance(result, TokiToolsResponse)
    assert result.tool_calls[0] is tc


async def test_acomplete_blocking_with_streaming_tool_call_prebuilt():
    """When `acomplete(stream=False)` returns a streaming tool, it should be an
    `AsyncTokiToolCallStream` whose `arguments()` and `expect_arg()` are pre-drained replays."""
    tc = TokiToolCall.from_dict({"id": "x", "function": {"name": "record", "arguments": json.dumps({"v": "abc"})}})
    model = _FakeAsyncModel(blocking_turn=_RawTurn(content="", tool_calls=[tc], thought=""))
    result = await model.acomplete(_msg(), tools=[StreamingToolSchema(schema={"type": "function", "function": {"name": "record"}})])
    assert isinstance(result, TokiToolsResponse)
    s = result.tool_calls[0]
    assert isinstance(s, AsyncTokiToolCallStream)
    arg = s.expect_arg("v")
    assert isinstance(arg, AsyncTokiArgStream)
    assert (await arg.value()) == "abc"


# ----- streaming ---------------------------------------------------------


async def test_acomplete_streaming_content_only():
    chunks: list[_RawChunk] = [_RawContentChunk(text="foo"), _RawContentChunk(text="bar")]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    out: list = []
    async for x in model.acomplete(_msg(), stream=True):
        out.append(x)
    assert out == ["foo", "bar"]


async def test_acomplete_streaming_thinking_yielded_when_capture_thinking():
    chunks: list[_RawChunk] = [
        _RawThoughtChunk(text="thinking..."),
        _RawContentChunk(text="answer"),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    out: list = []
    async for x in model.acomplete(_msg(), stream=True, capture_thinking=True):
        out.append(x)
    assert any(isinstance(x, TokiThinking) and x.text == "thinking..." for x in out)
    assert "answer" in out


async def test_acomplete_streaming_thinking_dropped_when_not_capture_thinking():
    chunks: list[_RawChunk] = [
        _RawThoughtChunk(text="ignore me"),
        _RawContentChunk(text="answer"),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    out: list = []
    async for x in model.acomplete(_msg(), stream=True, capture_thinking=False):
        out.append(x)
    assert out == ["answer"]


async def test_acomplete_streaming_static_tool_call_finalized():
    tools = [ToolSchema(schema={"type": "function", "function": {"name": "record"}})]
    chunks: list[_RawChunk] = [
        _RawToolCallChunk(index=0, id="t1", name="record"),
        _RawToolCallChunk(index=0, arguments_fragment='{"v": '),
        _RawToolCallChunk(index=0, arguments_fragment='42}'),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    out: list = []
    async for x in model.acomplete(_msg(), stream=True, tools=tools):
        out.append(x)
    tcs = [x for x in out if isinstance(x, TokiToolCall)]
    assert len(tcs) == 1
    assert tcs[0].function.arguments == {"v": 42}


async def test_acomplete_streaming_streaming_tool_expect_arg():
    tools = [StreamingToolSchema(schema={"type": "function", "function": {"name": "record"}})]
    chunks: list[_RawChunk] = [
        _RawToolCallChunk(index=0, id="t1", name="record"),
        _RawToolCallChunk(index=0, arguments_fragment='{"text": "hel'),
        _RawToolCallChunk(index=0, arguments_fragment='lo"}'),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    pieces: list[str] = []
    async for x in model.acomplete(_msg(), stream=True, tools=tools):
        if isinstance(x, AsyncTokiToolCallStream):
            arg = x.expect_arg("text")
            async for piece in arg:
                pieces.append(piece)
            assert (await x.arguments()) == {"text": "hello"}
    assert "".join(pieces) == "hello"


async def test_acomplete_streaming_streaming_tool_items():
    tools = [StreamingToolSchema(schema={"type": "function", "function": {"name": "record"}})]
    chunks: list[_RawChunk] = [
        _RawToolCallChunk(index=0, id="t1", name="record"),
        _RawToolCallChunk(index=0, arguments_fragment='{"a": "x", "b": 1}'),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    seen: list[tuple[str, object]] = []
    async for x in model.acomplete(_msg(), stream=True, tools=tools):
        if isinstance(x, AsyncTokiToolCallStream):
            async for name, arg in x.items():
                seen.append((name, await arg.value()))
    assert seen == [("a", "x"), ("b", 1)]


async def test_acomplete_streaming_usage_recorded():
    from toki.model import TokiUsageMetadata
    chunks: list[_RawChunk] = [
        _RawContentChunk(text="ok"),
        _RawUsage(usage=TokiUsageMetadata(prompt_tokens=3, completion_tokens=1, total_tokens=4)),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    async for _ in model.acomplete(_msg(), stream=True):
        pass
    assert model._usage_metadata is not None
    assert model._usage_metadata.total_tokens == 4


# ----- agent -------------------------------------------------------------


async def test_aexecute_streaming_records_history():
    from toki import Agent, WithoutTools
    chunks: list[_RawChunk] = [_RawContentChunk(text="hello"), _RawContentChunk(text=" world")]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    agent: Agent[WithoutTools] = Agent(model)
    agent.add_user_message("hi")
    pieces: list[str] = []
    async for x in agent.aexecute(stream=True):
        if isinstance(x, str):
            pieces.append(x)
    assert "".join(pieces) == "hello world"
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].content == "hello world"


async def test_aexecute_streaming_streaming_tool_materializes_to_history():
    from toki import Agent, WithStreamingTools
    tools = [StreamingToolSchema(schema={"type": "function", "function": {"name": "record"}})]
    chunks: list[_RawChunk] = [
        _RawToolCallChunk(index=0, id="t1", name="record"),
        _RawToolCallChunk(index=0, arguments_fragment='{"v": "ok"}'),
    ]
    model = _FakeAsyncModel(streaming_chunks=chunks)
    agent: Agent[WithStreamingTools] = Agent(model, tools=tools)
    agent.add_user_message("hi")
    async for x in agent.aexecute(stream=True):
        if isinstance(x, AsyncTokiToolCallStream):
            _ = await x.arguments()
    last = agent.messages[-1]
    assert last.role == "assistant"
    assert last.tool_calls is not None and len(last.tool_calls) == 1
    assert last.tool_calls[0].function.arguments == {"v": "ok"}
