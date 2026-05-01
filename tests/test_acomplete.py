"""Cartesian backend coverage for `BaseModel.acomplete()`.

Mirrors `test_complete.py` for the async surface. Same parameter axes and
sentinel-arg verification, but driven via `await` / `async for` and the
`AsyncTokiToolCallStream` wrapper.
"""

import pytest

from toki import (
    AsyncTokiToolCallStream,
    TokiThoughtResponse,
    TokiToolCall,
    TokiToolsResponse,
    TokiToolsThoughtResponse,
)

from .conftest import (
    SENTINEL_TEXT,
    SENTINEL_VALUE,
    adrain_stream,
    make_model,
    make_static_schema,
    make_streaming_schema,
)


PROVIDERS = ["openrouter", "openai", "anthropic", "google", "local", "ollama"]
TOOLS_SHAPES = ["none", "static", "streaming", "mixed"]


@pytest.mark.parametrize("tools_shape", TOOLS_SHAPES)
@pytest.mark.parametrize("capture_thinking", [False, True], ids=["nothink", "think"])
@pytest.mark.parametrize("stream", [False, True], ids=["block", "stream"])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_acomplete(provider: str, stream: bool, capture_thinking: bool, tools_shape: str):
    model = make_model(provider, reasoning=capture_thinking)

    if tools_shape == "none":
        await _run_no_tools(model, stream=stream, capture_thinking=capture_thinking)
    elif tools_shape == "static":
        await _run_static_tool(model, stream=stream, capture_thinking=capture_thinking)
    elif tools_shape == "streaming":
        await _run_streaming_tool(model, stream=stream, capture_thinking=capture_thinking)
    elif tools_shape == "mixed":
        await _run_mixed_tools(model, stream=stream, capture_thinking=capture_thinking)
    else:
        raise AssertionError(f"unknown tools_shape: {tools_shape!r}")


# ----- per-shape implementations -------------------------------------------

async def _run_no_tools(model, *, stream: bool, capture_thinking: bool) -> None:
    messages = [{
        "role": "user",
        "content": "Reply with exactly the word 'hello' and nothing else.",
    }]

    if not stream:
        result = await model.acomplete(messages, capture_thinking=capture_thinking)
        if capture_thinking:
            assert isinstance(result, TokiThoughtResponse), f"expected TokiThoughtResponse, got {type(result).__name__}"
            assert isinstance(result.content, str) and len(result.content) >= 1
            assert isinstance(result.thought, str) and len(result.thought) >= 1, "expected non-empty thought"
        else:
            assert isinstance(result, str), f"expected str, got {type(result).__name__}"
            assert len(result) >= 1
        return

    agen = model.acomplete(messages, stream=True, capture_thinking=capture_thinking)
    drained = await adrain_stream(agen)
    assert len(drained["strings"]) >= 1, "expected at least one content chunk"
    if capture_thinking:
        assert len(drained["thoughts"]) >= 1, "expected at least one TokiThinking chunk"


async def _run_static_tool(model, *, stream: bool, capture_thinking: bool) -> None:
    tool_name = "record_value"
    arg_name = "value"
    tools = [make_static_schema(tool_name, arg_name)]
    messages = [{
        "role": "user",
        "content": (
            f'Call the {tool_name} tool with {arg_name}="{SENTINEL_VALUE}". '
            "Do not respond with any other text."
        ),
    }]

    if not stream:
        result = await model.acomplete(messages, tools=tools, capture_thinking=capture_thinking)
        if capture_thinking:
            assert isinstance(result, (TokiToolsThoughtResponse, TokiThoughtResponse))
            if isinstance(result, TokiThoughtResponse):
                pytest.fail("model declined to call the tool")
            assert isinstance(result.thought, str) and len(result.thought) >= 1
        else:
            assert isinstance(result, TokiToolsResponse), (
                f"expected TokiToolsResponse, got {type(result).__name__}: {result!r}"
            )
        assert len(result.tool_calls) >= 1
        tc = result.tool_calls[0]
        assert isinstance(tc, TokiToolCall), f"expected TokiToolCall, got {type(tc).__name__}"
        assert tc.function.name == tool_name
        assert tc.function.arguments == {arg_name: SENTINEL_VALUE}
        return

    agen = model.acomplete(messages, stream=True, tools=tools, capture_thinking=capture_thinking)
    drained = await adrain_stream(agen)
    assert len(drained["tool_calls"]) >= 1, "expected at least one TokiToolCall in stream"
    tc = drained["tool_calls"][0]
    assert tc.function.name == tool_name
    assert tc.function.arguments == {arg_name: SENTINEL_VALUE}
    if capture_thinking:
        assert len(drained["thoughts"]) >= 1, "expected at least one TokiThinking chunk"


async def _run_streaming_tool(model, *, stream: bool, capture_thinking: bool) -> None:
    tool_name = "record_value"
    arg_name = "value"
    tools = [make_streaming_schema(tool_name, arg_name)]
    messages = [{
        "role": "user",
        "content": (
            f'Call the {tool_name} tool with {arg_name}="{SENTINEL_VALUE}". '
            "Do not respond with any other text."
        ),
    }]

    if not stream:
        result = await model.acomplete(messages, tools=tools, capture_thinking=capture_thinking)
        if isinstance(result, (str, TokiThoughtResponse)):
            pytest.fail(f"model declined to call the tool: {result!r}")
        if capture_thinking:
            assert isinstance(result, TokiToolsThoughtResponse)
            assert isinstance(result.thought, str) and len(result.thought) >= 1
        else:
            assert isinstance(result, TokiToolsResponse)
        assert len(result.tool_calls) >= 1
        tc = result.tool_calls[0]
        assert isinstance(tc, AsyncTokiToolCallStream), f"expected AsyncTokiToolCallStream, got {type(tc).__name__}"
        assert tc.name == tool_name
        assert (await tc.arguments()) == {arg_name: SENTINEL_VALUE}
        return

    agen = model.acomplete(messages, stream=True, tools=tools, capture_thinking=capture_thinking)
    drained = await adrain_stream(agen, expected_arg_names={tool_name: arg_name})
    assert len(drained["tool_streams"]) >= 1, "expected at least one AsyncTokiToolCallStream"
    s = drained["tool_streams"][0]
    assert s.name == tool_name
    assert (await s.arguments()) == {arg_name: SENTINEL_VALUE}
    if capture_thinking:
        assert len(drained["thoughts"]) >= 1, "expected at least one TokiThinking chunk"


async def _run_mixed_tools(model, *, stream: bool, capture_thinking: bool) -> None:
    static_name = "record_static"
    streaming_name = "record_streaming"
    tools = [
        make_static_schema(static_name, "value"),
        make_streaming_schema(streaming_name, "text"),
    ]
    messages = [{
        "role": "user",
        "content": (
            f'Call the {static_name} tool with value="{SENTINEL_VALUE}" '
            f'AND call the {streaming_name} tool with text="{SENTINEL_TEXT}". '
            "Make both calls in parallel in the same response. Do not respond with any other text."
        ),
    }]

    if not stream:
        result = await model.acomplete(messages, tools=tools, capture_thinking=capture_thinking)
        if isinstance(result, (str, TokiThoughtResponse)):
            pytest.fail("model declined to call any tools")
        assert isinstance(result, (TokiToolsResponse, TokiToolsThoughtResponse))
        if capture_thinking:
            assert isinstance(result, TokiToolsThoughtResponse)
            assert isinstance(result.thought, str) and len(result.thought) >= 1
        await _assert_mixed_calls(result.tool_calls, static_name, streaming_name)
        return

    agen = model.acomplete(messages, stream=True, tools=tools, capture_thinking=capture_thinking)
    drained = await adrain_stream(agen, expected_arg_names={streaming_name: "text"})
    combined = drained["tool_calls"] + drained["tool_streams"]
    await _assert_mixed_calls(combined, static_name, streaming_name)
    if capture_thinking:
        assert len(drained["thoughts"]) >= 1, "expected at least one TokiThinking chunk"


async def _assert_mixed_calls(tool_calls: list, static_name: str, streaming_name: str) -> None:
    """Locate one TokiToolCall (static) and one AsyncTokiToolCallStream (streaming),
    each with the expected sentinel argument."""
    static_calls = [tc for tc in tool_calls if isinstance(tc, TokiToolCall) and tc.function.name == static_name]
    stream_calls = [tc for tc in tool_calls if isinstance(tc, AsyncTokiToolCallStream) and tc.name == streaming_name]
    assert len(static_calls) >= 1, f"missing static tool call {static_name!r} (got {tool_calls!r})"
    assert len(stream_calls) >= 1, f"missing streaming tool call {streaming_name!r} (got {tool_calls!r})"
    assert static_calls[0].function.arguments == {"value": SENTINEL_VALUE}
    assert (await stream_calls[0].arguments()) == {"text": SENTINEL_TEXT}
