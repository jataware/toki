"""`Agent.aexecute` orchestration tests.

Async mirror of `test_agent.py`. Pinned to the same single cheap backend (Google
default) to keep the matrix small while exercising the orchestration shapes
`Agent` adds on top of `BaseModel`: history tracking, static + streaming tool
round-trips, thought-exclusion from history, and the four type-parameter overloads.
"""

import pytest

from toki import (
    Agent,
    AsyncTokiToolCallStream,
    TokiToolCall,
    TokiToolsResponse,
    WithMixedTools,
    WithoutTools,
    WithStaticTools,
    WithStreamingTools,
)

from .conftest import (
    SENTINEL_VALUE,
    adrain_stream,
    make_model,
    make_static_schema,
    make_streaming_schema,
)


def _agent_model():
    return make_model("google", reasoning=False)


def _reasoning_agent_model():
    return make_model("google", reasoning=True)


async def test_aagent_simple_round_trip():
    """Two-turn no-tool conversation; assert history grows correctly."""
    agent: Agent[WithoutTools] = Agent(_agent_model())
    agent.add_user_message("Reply with the single word 'one'.")
    first = await agent.aexecute()
    assert isinstance(first, str) and len(first) >= 1

    agent.add_user_message("Now reply with the single word 'two'.")
    second = await agent.aexecute()
    assert isinstance(second, str) and len(second) >= 1

    assert len(agent.messages) == 4
    assert [m.role for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert agent.messages[1].content == first
    assert agent.messages[3].content == second


async def test_aagent_static_tool_round_trip():
    """Static tool round-trip: model calls tool, we add a tool result, model replies."""
    tool = make_static_schema("record_value", "value")
    agent: Agent[WithStaticTools] = Agent(_agent_model(), tools=[tool])
    agent.add_user_message(
        f'Please call the record_value tool with value="{SENTINEL_VALUE}".'
    )

    first = await agent.aexecute()
    assert isinstance(first, TokiToolsResponse), f"expected TokiToolsResponse, got {type(first).__name__}"
    tc = first.tool_calls[0]
    assert isinstance(tc, TokiToolCall)
    assert tc.function.name == "record_value"
    assert tc.function.arguments == {"value": SENTINEL_VALUE}

    agent.add_tool_message(tc.id, "ok")
    final = await agent.aexecute()
    assert isinstance(final, str) and len(final) >= 1

    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.messages[1].tool_calls is not None and len(agent.messages[1].tool_calls) >= 1
    assert agent.messages[2].tool_call_id == tc.id


async def test_aagent_streaming_tool_round_trip():
    """Streaming tool round-trip via `aexecute(stream=True)`. After draining the
    `AsyncTokiToolCallStream`, history should contain a materialized `TokiToolCall`
    with the expected arguments."""
    tool = make_streaming_schema("record_value", "value")
    agent: Agent[WithStreamingTools] = Agent(_agent_model(), tools=[tool])
    agent.add_user_message(
        f'Please call the record_value tool with value="{SENTINEL_VALUE}".'
    )

    drained = await adrain_stream(agent.aexecute(stream=True), expected_arg_names={"record_value": "value"})
    assert len(drained["tool_streams"]) >= 1
    s = drained["tool_streams"][0]
    assert (await s.arguments()) == {"value": SENTINEL_VALUE}

    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].tool_calls is not None and len(agent.messages[-1].tool_calls) == 1
    rebuilt = agent.messages[-1].tool_calls[0]
    assert isinstance(rebuilt, TokiToolCall)
    assert rebuilt.function.arguments == {"value": SENTINEL_VALUE}

    agent.add_tool_message(rebuilt.id, "ok")
    final = await agent.aexecute()
    assert isinstance(final, str) and len(final) >= 1


async def test_aagent_capture_thinking_excluded_from_history():
    """`TokiThinking` fragments must not be persisted to history when
    `capture_thinking=True`. The assistant message stored should be just the
    visible content text."""
    agent: Agent[WithoutTools] = Agent(_reasoning_agent_model())
    agent.add_user_message("Reply with exactly the word 'hi' and nothing else.")
    result = await agent.aexecute(capture_thinking=True)

    assert hasattr(result, "thought") and isinstance(result.thought, str) and len(result.thought) >= 1
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].content == result.content
    assert result.thought not in agent.messages[-1].content
    assert not hasattr(agent.messages[-1], "thought")


def test_aagent_type_parameters():
    """Smoke test for the four `Agent[...]` overloads; same as the sync version
    since `Agent` construction itself doesn't differ between sync/async."""
    static_tool = make_static_schema("static_record", "value")
    streaming_tool = make_streaming_schema("streaming_record", "text")

    m = _agent_model()

    a_none: Agent[WithoutTools] = Agent(m)
    a_static: Agent[WithStaticTools] = Agent(m, tools=[static_tool])
    a_streaming: Agent[WithStreamingTools] = Agent(m, tools=[streaming_tool])
    a_mixed: Agent[WithMixedTools] = Agent(m, tools=[static_tool, streaming_tool])

    assert a_none.tools is None
    assert a_static.tools is not None and len(a_static.tools) == 1
    assert a_streaming.tools is not None and len(a_streaming.tools) == 1
    assert a_mixed.tools is not None and len(a_mixed.tools) == 2
    for a in (a_none, a_static, a_streaming, a_mixed):
        assert a.messages == []
