import importlib
import os
import sys
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from toki import (
    Agent,
    ClaudeAdaptiveReasoning,
    ClaudeBudgetReasoning,
    NovaReasoning,
    StreamingToolSchema,
    TokiCacheWarning,
    TokiMessage,
    TokiThinking,
    TokiThoughtResponse,
    TokiToolCall,
    TokiToolFunction,
    TokiToolsResponse,
    ToolSchema,
    WithStaticTools,
)

from .conftest import make_static_schema


def _text_response(text: str = "hello") -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 3,
            "outputTokens": 2,
            "totalTokens": 5,
        },
    }


def _reasoning_tool_response() -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "I should call the tool.",
                                "signature": "signed-value",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": "record_value",
                            "input": {"value": "apple"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 8,
            "outputTokens": 4,
            "totalTokens": 12,
        },
    }


def _reasoning_text_response(text: str = "answer") -> dict:
    response = _reasoning_tool_response()
    response["output"]["message"]["content"][1] = {"text": text}
    response["stopReason"] = "end_turn"
    return response


class FakeEventStream:
    def __init__(self, events: list[dict]):
        self._events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return next(self._events)

    def close(self) -> None:
        self.closed = True


class FakeBedrockClient:
    def __init__(self):
        self.client_creation_calls: list[dict] = []
        self.converse_calls: list[dict] = []
        self.converse_stream_calls: list[dict] = []
        self.count_tokens_calls: list[dict] = []
        self.converse_responses: list[dict] = [_text_response()]
        self.streams: list[FakeEventStream] = []

    def converse(self, **kwargs) -> dict:
        self.converse_calls.append(kwargs)
        return self.converse_responses.pop(0)

    def converse_stream(self, **kwargs) -> dict:
        self.converse_stream_calls.append(kwargs)
        return {"stream": self.streams.pop(0)}

    def count_tokens(self, **kwargs) -> dict:
        self.count_tokens_calls.append(kwargs)
        return {"inputTokens": 17}


@pytest.fixture
def bedrock(monkeypatch):
    client = FakeBedrockClient()
    sessions: list[dict] = []

    def Session(**kwargs):
        sessions.append(kwargs)

        def make_client(service: str, **client_kwargs):
            client.client_creation_calls.append(
                {
                    "service": service,
                    **client_kwargs,
                }
            )
            return client

        return SimpleNamespace(client=make_client)

    for name in ["toki.bedrock.model", "toki.bedrock"]:
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=Session))
    module = importlib.import_module("toki.bedrock.model")
    yield module, client, sessions
    for name in ["toki.bedrock.model", "toki.bedrock"]:
        sys.modules.pop(name, None)


def test_constructor_uses_profile_region_and_optional_api_key(
    bedrock,
    monkeypatch,
):
    module, _, sessions = bedrock
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    model = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
        api_key="bedrock-key",
        profile_name="work",
        region_name="us-east-1",
    )

    assert model.model == "global.anthropic.claude-sonnet-4-6"
    assert sessions == [{"profile_name": "work", "region_name": "us-east-1"}]
    assert model._client is bedrock[1]
    assert module.os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-key"
    config = bedrock[1].client_creation_calls[0]["config"]
    assert config.auth_scheme_preference == "httpBearerAuth"


def test_constructor_leaves_existing_auth_chain_untouched(bedrock, monkeypatch):
    module, _, _ = bedrock
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "existing")

    module.BedrockModel("amazon.nova-pro-v1:0")

    assert module.os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "existing"
    config = bedrock[1].client_creation_calls[0]["config"]
    assert config.auth_scheme_preference == "httpBearerAuth"


def test_constructor_uses_default_sigv4_without_bearer_token(
    bedrock,
    monkeypatch,
):
    module, client, _ = bedrock
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    module.BedrockModel("amazon.nova-pro-v1:0")

    assert client.client_creation_calls == [{"service": "bedrock-runtime"}]


def test_blocking_request_converts_messages_tools_and_usage(bedrock):
    module, client, _ = bedrock
    model = module.BedrockModel("model-id", allow_parallel_tool_calls=True)
    prior_call = TokiToolCall(
        id="old-call",
        function=TokiToolFunction(name="record_value", arguments={"value": "old"}),
    )
    messages = [
        TokiMessage(role="system", content="Be concise."),
        TokiMessage(role="user", content="Use the tool."),
        TokiMessage(role="assistant", content="", tool_calls=[prior_call]),
        TokiMessage(role="tool", content="first result", tool_call_id="old-call"),
        TokiMessage(role="tool", content="second result", tool_call_id="other-call"),
        TokiMessage(role="user", content="Continue."),
    ]

    result = model.complete(
        messages,
        tools=[make_static_schema("record_value", "value")],
        inferenceConfig={"maxTokens": 100},
    )

    assert result == "hello"
    request = client.converse_calls[0]
    assert request["modelId"] == "model-id"
    assert request["system"] == [{"text": "Be concise."}]
    assert request["inferenceConfig"] == {"maxTokens": 100}
    assert request["toolConfig"]["tools"][0]["toolSpec"] == {
        "name": "record_value",
        "description": "Record a value for testing.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to record.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            }
        },
    }
    assert request["messages"][2] == {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "old-call",
                    "content": [{"text": "first result"}],
                }
            },
            {
                "toolResult": {
                    "toolUseId": "other-call",
                    "content": [{"text": "second result"}],
                }
            },
        ],
    }
    assert model._usage_metadata is not None
    assert model._usage_metadata.total_tokens == 5


def test_agent_preserves_and_replays_reasoning_for_tool_result(bedrock):
    module, client, _ = bedrock
    client.converse_responses = [_reasoning_tool_response(), _text_response("done")]
    model = module.BedrockModel("model-id")
    agent: Agent[WithStaticTools] = Agent(
        model,
        tools=[make_static_schema("record_value", "value")],
    )
    agent.add_user_message("Record apple.")

    first = agent.execute()

    assert isinstance(first, TokiToolsResponse)
    call = first.tool_calls[0]
    assert isinstance(call, TokiToolCall)
    state = agent.messages[-1].provider_state
    assert state == {
        "bedrock": {
            "reasoning_content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "I should call the tool.",
                            "signature": "signed-value",
                        }
                    }
                }
            ]
        }
    }

    agent.add_tool_message(call.id, "ok")
    assert agent.execute() == "done"

    assistant = client.converse_calls[1]["messages"][1]
    assert list(assistant["content"][0]) == ["reasoningContent"]
    assert (
        assistant["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        == "signed-value"
    )
    assert list(assistant["content"][1]) == ["toolUse"]


def test_agent_preserves_reasoning_without_tool_calls(bedrock):
    module, client, _ = bedrock
    client.converse_responses = [
        _reasoning_text_response(),
        _text_response("second"),
    ]
    model = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
    )
    agent = Agent(model)
    agent.add_user_message("Think.")

    first = agent.execute(capture_thinking=True)

    assert isinstance(first, TokiThoughtResponse)
    assert first.thought == "I should call the tool."
    assert agent.messages[-1].provider_state is not None

    agent.add_user_message("Continue.")
    assert agent.execute() == "second"
    replayed = client.converse_calls[1]["messages"][1]["content"]
    assert list(replayed[0]) == ["reasoningContent"]
    assert replayed[1] == {"text": "answer"}


def test_hidden_reasoning_still_preserves_provider_state(bedrock):
    module, client, _ = bedrock
    client.converse_responses = [_reasoning_text_response()]
    model = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
        reasoning_effort="medium",
    )
    agent = Agent(model)
    agent.add_user_message("Think privately.")

    assert agent.execute() == "answer"
    assert agent.messages[-1].provider_state is not None


def test_reasoning_effort_and_typed_configs_map_by_family(bedrock):
    module, client, _ = bedrock
    client.converse_responses = [
        _text_response(),
        _text_response(),
        _text_response(),
    ]

    claude = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
        reasoning_effort="high",
    )
    claude.complete([{"role": "user", "content": "Hi"}])
    assert client.converse_calls[-1]["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    budget = module.BedrockModel(
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        reasoning_config=ClaudeBudgetReasoning(budget_tokens=2_000),
    )
    budget.complete([{"role": "user", "content": "Hi"}])
    assert client.converse_calls[-1]["inferenceConfig"]["maxTokens"] == 3_024

    nova = module.BedrockModel(
        "us.amazon.nova-2-lite-v1:0",
        reasoning_config=NovaReasoning(effort="low"),
    )
    nova.complete([{"role": "user", "content": "Hi"}])
    assert client.converse_calls[-1]["additionalModelRequestFields"] == {
        "reasoningConfig": {
            "type": "enabled",
            "maxReasoningEffort": "low",
        }
    }

    with pytest.raises(ValueError):
        module.BedrockModel(
            "global.anthropic.claude-sonnet-4-6",
            reasoning_effort="medium",
            reasoning_config=ClaudeAdaptiveReasoning(),
        )


def _stream_events() -> list[dict]:
    return [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "reasoningContent": {"text": "Call the "},
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "reasoningContent": {
                        "text": "tool.",
                        "signature": "stream-signature",
                    },
                },
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {
                    "toolUse": {
                        "toolUseId": "stream-call",
                        "name": "record_value",
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"value":"app'}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": 'le"}'}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 6,
                    "totalTokens": 16,
                }
            }
        },
    ]


def _stream_reasoning_text_events() -> list[dict]:
    return [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "reasoningContent": {
                        "text": "Think.",
                        "signature": "text-signature",
                    }
                },
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"text": "answer"},
            }
        },
    ]


def test_streaming_translates_thinking_tool_deltas_and_usage(bedrock):
    module, client, _ = bedrock
    event_stream = FakeEventStream(_stream_events())
    client.streams = [event_stream]
    model = module.BedrockModel("model-id")
    tool = ToolSchema(schema=make_static_schema("record_value", "value").schema)

    with pytest.warns():
        chunks = list(
            model.complete(
                [{"role": "user", "content": "Record apple."}],
                stream=True,
                tools=[tool],
                capture_thinking=True,
            )
        )

    thoughts = [chunk.text for chunk in chunks if isinstance(chunk, TokiThinking)]
    calls = [chunk for chunk in chunks if isinstance(chunk, TokiToolCall)]
    assert thoughts == ["Call the ", "tool."]
    assert len(calls) == 1
    assert calls[0].function.arguments == {"value": "apple"}
    assert (
        calls[0].provider_state["bedrock"]["reasoning_content"][0]["reasoningContent"][
            "reasoningText"
        ]["signature"]
        == "stream-signature"
    )
    assert event_stream.closed
    assert model._usage_metadata is not None
    assert model._usage_metadata.total_tokens == 16


def test_streaming_tool_schema_receives_provider_state(bedrock):
    module, client, _ = bedrock
    client.streams = [FakeEventStream(_stream_events())]
    model = module.BedrockModel("model-id")
    tool = StreamingToolSchema(
        schema=make_static_schema("record_value", "value").schema
    )

    stream = model.complete(
        [{"role": "user", "content": "Record apple."}],
        stream=True,
        tools=[tool],
    )
    call_stream = next(chunk for chunk in stream if not isinstance(chunk, str))

    assert call_stream.provider_state is not None
    assert call_stream.arguments == {"value": "apple"}


def test_streaming_agent_lifts_provider_state_to_message(bedrock):
    module, client, _ = bedrock
    client.streams = [FakeEventStream(_stream_events())]
    model = module.BedrockModel("model-id")
    tool = StreamingToolSchema(
        schema=make_static_schema("record_value", "value").schema
    )
    agent = Agent(model, tools=[tool])
    agent.add_user_message("Record apple.")

    chunks = list(agent.execute(stream=True))

    assert any(hasattr(chunk, "arguments") for chunk in chunks)
    assert agent.messages[-1].provider_state is not None
    assert (
        agent.messages[-1].provider_state["bedrock"]["reasoning_content"][0][
            "reasoningContent"
        ]["reasoningText"]["signature"]
        == "stream-signature"
    )


def test_count_tokens_uses_converse_shape(bedrock):
    module, client, _ = bedrock
    model = module.BedrockModel("model-id")

    count = model.count_tokens(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}],
        tools=[make_static_schema("record_value", "value")],
    )

    assert count == 17
    assert client.count_tokens_calls[0]["modelId"] == "model-id"
    converse = client.count_tokens_calls[0]["input"]["converse"]
    assert converse["system"] == [{"text": "System"}]
    assert converse["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert "toolConfig" in converse
    with pytest.raises(ValueError):
        model.count_tokens([], kind="offline")


def test_cache_usage_tool_choice_and_static_cache_point(bedrock):
    module, client, _ = bedrock
    response = _text_response()
    response["usage"].update(
        {
            "cacheReadInputTokens": 100,
            "cacheWriteInputTokens": 20,
        }
    )
    client.converse_responses = [response]
    schema = make_static_schema("record_value", "value").schema
    schema["function"]["strict"] = True
    model = module.BedrockModel(
        "amazon.nova-micro-v1:0",
        cache="static",
    )

    model.complete(
        [{"role": "user", "content": "x" * 5_000}],
        tools=[schema],
        toolConfig={"toolChoice": {"any": {}}},
    )

    request = client.converse_calls[0]
    assert request["messages"][0]["content"][-1] == {"cachePoint": {"type": "default"}}
    assert request["toolConfig"]["toolChoice"] == {"any": {}}
    assert "strict" not in request["toolConfig"]["tools"][0]["toolSpec"]
    assert model._usage_metadata is not None
    assert model._usage_metadata.prompt_tokens == 123
    assert model._usage_metadata.total_tokens == 125
    assert model._usage_metadata.cache_read_tokens == 100
    assert model._usage_metadata.cache_write_tokens == 20


def test_unsupported_explicit_cache_warns_and_sends_no_marker(bedrock):
    module, client, _ = bedrock
    model = module.BedrockModel(
        "global.openai.gpt-5.6-luna",
        cache="static",
    )

    with pytest.warns(TokiCacheWarning):
        model.complete([{"role": "user", "content": "x" * 5_000}])

    assert client.converse_calls[0]["messages"][0]["content"] == [{"text": "x" * 5_000}]


def test_rolling_cache_advances_to_previous_turn(bedrock):
    module, client, _ = bedrock
    model = module.BedrockModel(
        "amazon.nova-micro-v1:0",
        cache="rolling",
    )
    messages = [
        {"role": "user", "content": "x" * 5_000},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    model.complete(messages)

    wire = client.converse_calls[0]["messages"]
    assert wire[1]["content"][-1] == {"cachePoint": {"type": "default"}}
    assert wire[2]["content"] == [{"text": "second"}]
    assert model._cache_state.entries
    model.invalidate_cache()
    assert model._cache_state.entries == []


def test_count_tokens_profile_falls_back_to_converse(bedrock):
    module, client, _ = bedrock

    def unsupported(**kwargs):
        client.count_tokens_calls.append(kwargs)
        raise ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "unsupported profile",
                }
            },
            "CountTokens",
        )

    client.count_tokens = unsupported
    model = module.BedrockModel("us.amazon.nova-2-lite-v1:0")

    with pytest.warns(match="one-output-token Converse"):
        count = model.count_tokens(
            [{"role": "user", "content": "Hi"}],
            kind="online",
        )

    assert count == 3
    assert client.converse_calls[0]["inferenceConfig"] == {"maxTokens": 1}


def test_count_tokens_claude_profile_uses_mantle(
    bedrock,
    monkeypatch,
):
    module, client, _ = bedrock

    def unsupported(**kwargs):
        raise ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "unsupported profile",
                }
            },
            "CountTokens",
        )

    captured: dict = {}

    def mantle(**kwargs):
        captured.update(kwargs)
        return 41

    client.count_tokens = unsupported
    monkeypatch.setattr(module, "count_claude_tokens", mantle)
    model = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
        region_name="us-east-1",
    )

    count = model.count_tokens([{"role": "user", "content": "Hi"}])

    assert count == 41
    assert captured["region_name"] == "us-east-1"
    assert captured["model_id"] == "global.anthropic.claude-sonnet-4-6"


async def test_async_completion_streaming_and_counting(bedrock):
    module, client, _ = bedrock
    client.converse_responses = [_text_response("async")]
    client.streams = [
        FakeEventStream(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "stream"},
                    }
                }
            ]
        )
    ]
    model = module.BedrockModel("model-id")

    assert await model.acomplete([{"role": "user", "content": "Hi"}]) == "async"
    chunks = [
        chunk
        async for chunk in model.acomplete(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        )
    ]
    assert chunks == ["stream"]
    assert await model.acount_tokens([{"role": "user", "content": "Hi"}]) == 17


async def test_async_agent_preserves_streamed_reasoning_without_tools(bedrock):
    module, client, _ = bedrock
    client.streams = [FakeEventStream(_stream_reasoning_text_events())]
    model = module.BedrockModel(
        "global.anthropic.claude-sonnet-4-6",
    )
    agent = Agent(model)
    agent.add_user_message("Think.")

    chunks = [
        chunk
        async for chunk in agent.aexecute(
            stream=True,
            capture_thinking=True,
        )
    ]

    assert [chunk.text for chunk in chunks if isinstance(chunk, TokiThinking)] == [
        "Think."
    ]
    assert agent.messages[-1].content == "answer"
    assert agent.messages[-1].provider_state is not None
    assert (
        agent.messages[-1].provider_state["bedrock"]["reasoning_content"][0][
            "reasoningContent"
        ]["reasoningText"]["signature"]
        == "text-signature"
    )


def test_stream_error_is_raised(bedrock):
    module, client, _ = bedrock
    client.streams = [
        FakeEventStream(
            [
                {"throttlingException": {"message": "slow down"}},
            ]
        )
    ]
    model = module.BedrockModel("model-id")

    with pytest.raises(RuntimeError, match="throttlingException: slow down"):
        list(
            model.complete(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )


@pytest.mark.cost_integration
def test_bedrock_live_smoke():
    model_id = os.getenv("BEDROCK_TEST_MODEL")
    if model_id is None:
        pytest.fail(
            "BEDROCK_TEST_MODEL is not set; required for the Bedrock smoke test"
        )

    from toki import BedrockModel

    model = BedrockModel(model_id)
    result = model.complete(
        [
            {"role": "user", "content": "Reply with exactly the word hello."},
        ]
    )
    assert isinstance(result, str)
    assert result


@pytest.mark.cost_integration
def test_bedrock_live_reasoning():
    model_id = os.getenv("BEDROCK_REASONING_TEST_MODEL")
    if model_id is None:
        pytest.fail(
            "BEDROCK_REASONING_TEST_MODEL is not set; required for the "
            "Bedrock reasoning test"
        )

    from toki import BedrockModel

    model = BedrockModel(model_id, reasoning_effort="medium")
    result = model.complete(
        [{"role": "user", "content": "What is 17 times 19?"}],
        capture_thinking=True,
    )
    assert isinstance(result, TokiThoughtResponse)
    assert result.content
    assert result.thought
    assert model.last_provider_state is not None


@pytest.mark.cost_integration
def test_bedrock_live_tool_call():
    model_id = os.getenv("BEDROCK_TOOL_TEST_MODEL")
    if model_id is None:
        pytest.fail(
            "BEDROCK_TOOL_TEST_MODEL is not set; required for the Bedrock tool test"
        )

    from toki import BedrockModel

    model = BedrockModel(model_id)
    result = model.complete(
        [{"role": "user", "content": "Record the value apple."}],
        tools=[make_static_schema("record_value", "value")],
        toolConfig={"toolChoice": {"any": {}}},
    )
    assert isinstance(result, TokiToolsResponse)
    assert result.tool_calls[0].function.name == "record_value"


@pytest.mark.cache_integration
def test_bedrock_live_static_cache():
    model_id = os.getenv("BEDROCK_CACHE_TEST_MODEL")
    if model_id is None:
        pytest.fail(
            "BEDROCK_CACHE_TEST_MODEL is not set; required for the Bedrock cache test"
        )

    from toki import BedrockModel
    from toki.bedrock.models import get_bedrock_model_attributes

    attributes = get_bedrock_model_attributes(model_id)
    if attributes is None or attributes.cache_min_tokens is None:
        pytest.fail(
            "BEDROCK_CACHE_TEST_MODEL must be present in the bundled cache metadata"
        )
    model = BedrockModel(model_id, cache="static")
    messages = [
        {
            "role": "user",
            "content": "Use this reference, then reply with OK:\n"
            + ("reference text " * attributes.cache_min_tokens),
        }
    ]

    first = model.complete(
        messages,
        inferenceConfig={"maxTokens": 10},
    )
    first_usage = model.usage_metadata
    assert isinstance(first, str)
    assert first_usage is not None
    assert first_usage.cache_write_tokens > 0

    messages.extend(
        [
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Reply with OK again."},
        ]
    )
    model.complete(messages, inferenceConfig={"maxTokens": 10})
    assert model.usage_metadata is not None
    assert model.usage_metadata.cache_read_tokens > 0
