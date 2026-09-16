"""Live OpenAI Responses coverage beyond the cartesian complete/acomplete matrix.

Requires OPENAI_API_KEY. Asserts the GPT-5.4 Completions hole is closed: tools
plus capture_thinking must not 400. Also checks a tool round-trip so
provider_state replayed on the next turn does not 400 either.
"""

import os

import pytest

from toki import (
    Agent,
    OpenAIResponsesModel,
    TokiThoughtResponse,
    TokiToolsResponse,
    TokiToolsThoughtResponse,
    WithStaticTools,
)

from .conftest import SENTINEL_VALUE, make_static_schema


def test_openai_responses_tools_plus_reasoning_does_not_400():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is not set; required to run openai_responses tests")
    model = OpenAIResponsesModel(
        "gpt-5.4-nano",
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort="medium",
        allow_parallel_tool_calls=True,
    )
    tools = [make_static_schema("record_value", "value")]
    result = model.complete(
        [{
            "role": "user",
            "content": (
                f'Call the record_value tool with value="{SENTINEL_VALUE}". '
                "Do not respond with any other text."
            ),
        }],
        tools=tools,
        capture_thinking=True,
    )
    assert isinstance(result, (TokiToolsResponse, TokiToolsThoughtResponse, TokiThoughtResponse, str))
    if isinstance(result, (TokiToolsResponse, TokiToolsThoughtResponse)):
        assert result.tool_calls[0].function.arguments == {"value": SENTINEL_VALUE}


def test_openai_responses_agent_tool_round_trip_with_reasoning():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is not set; required to run openai_responses tests")
    model = OpenAIResponsesModel(
        "gpt-5.4-nano",
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort="medium",
        allow_parallel_tool_calls=True,
    )
    tool = make_static_schema("record_value", "value")
    agent: Agent[WithStaticTools] = Agent(model, tools=[tool])
    agent.add_user_message(
        f'Please call the record_value tool with value="{SENTINEL_VALUE}".'
    )
    first = agent.execute(capture_thinking=True)
    if isinstance(first, (str, TokiThoughtResponse)):
        pytest.fail(f"model declined to call the tool: {first!r}")
    tc = first.tool_calls[0]
    assert tc.function.arguments == {"value": SENTINEL_VALUE}
    agent.add_tool_message(tc.id, "ok")
    final = agent.execute(capture_thinking=True)
    assert isinstance(final, (str, TokiThoughtResponse))
    if isinstance(final, TokiThoughtResponse):
        assert len(final.content) >= 1
    else:
        assert len(final) >= 1
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool", "assistant"]
    assert agent.messages[2].tool_call_id == tc.id
