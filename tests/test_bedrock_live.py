"""End-to-end Bedrock verification using real AWS credentials.

Run explicitly because this test makes paid inference calls:

    uv run pytest -m "cost_integration and cache_integration" \
        tests/test_bedrock_live.py -v

The defaults target models used by Toki's Bedrock examples. Override them with
BEDROCK_LIVE_REGION, BEDROCK_LIVE_MODEL, BEDROCK_LIVE_REASONING_MODEL, and
BEDROCK_LIVE_PROFILE when needed. Authentication comes from boto3's normal
credential chain or AWS_BEARER_TOKEN_BEDROCK.
"""

import asyncio
import os

import pytest

from toki import (
    BedrockModel,
    TokiThoughtResponse,
    TokiToolCall,
    TokiToolsResponse,
)
from toki.bedrock import (
    discover_bedrock_models,
    list_bedrock_models,
)
from toki.bedrock.models import get_bedrock_model_attributes

REGION = os.getenv("BEDROCK_LIVE_REGION", "us-east-2")
MODEL_ID = os.getenv(
    "BEDROCK_LIVE_MODEL",
    "us.amazon.nova-micro-v1:0",
)
REASONING_MODEL_ID = os.getenv(
    "BEDROCK_LIVE_REASONING_MODEL",
    "global.anthropic.claude-sonnet-4-6",
)
PROFILE = os.getenv("BEDROCK_LIVE_PROFILE")

RECORD_VALUE = {
    "type": "function",
    "function": {
        "name": "record_value",
        "description": "Record a string value.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The value to record.",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def _model(model_id: str, **kwargs) -> BedrockModel:
    return BedrockModel(
        model_id,
        profile_name=PROFILE,
        region_name=REGION,
        **kwargs,
    )


async def _verify_async(model: BedrockModel) -> None:
    response = await model.acomplete(
        [{"role": "user", "content": "Reply with exactly: async"}],
        inferenceConfig={"maxTokens": 10},
    )
    assert "async" in response.lower()

    chunks = [
        chunk
        async for chunk in model.acomplete(
            [{"role": "user", "content": "Reply with exactly: stream"}],
            stream=True,
            inferenceConfig={"maxTokens": 10},
        )
    ]
    assert "".join(chunk for chunk in chunks if isinstance(chunk, str)).strip()


@pytest.mark.cost_integration
@pytest.mark.cache_integration
def test_bedrock_live_feature_parity() -> None:
    static_models = list_bedrock_models()
    assert MODEL_ID in static_models or get_bedrock_model_attributes(MODEL_ID)

    discovered = discover_bedrock_models(
        profile_name=PROFILE,
        region_name=REGION,
        refresh=True,
    )
    assert discovered
    assert any(info.kind == "foundation_model" for info in discovered)
    assert any(info.kind == "inference_profile" for info in discovered)

    model = _model(MODEL_ID, allow_parallel_tool_calls=True)
    response = model.complete(
        [{"role": "user", "content": "Reply with exactly: bedrock"}],
        inferenceConfig={"maxTokens": 10},
    )
    assert "bedrock" in response.lower()
    assert model.usage_metadata is not None
    assert model.usage_metadata.total_tokens > 0

    chunks = list(
        model.complete(
            [{"role": "user", "content": "Reply with exactly: streaming"}],
            stream=True,
            inferenceConfig={"maxTokens": 10},
        )
    )
    assert "".join(chunk for chunk in chunks if isinstance(chunk, str)).strip()

    asyncio.run(_verify_async(model))

    token_count = model.count_tokens(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Count this prompt."},
        ],
        tools=[RECORD_VALUE],
        kind="online",
    )
    assert token_count > 0

    tool_response = model.complete(
        [{"role": "user", "content": "Record the value apple using the tool."}],
        tools=[RECORD_VALUE],
        toolConfig={"toolChoice": {"any": {}}},
        inferenceConfig={"maxTokens": 50},
    )
    assert isinstance(tool_response, TokiToolsResponse)
    assert len(tool_response.tool_calls) == 1
    tool_call = tool_response.tool_calls[0]
    assert isinstance(tool_call, TokiToolCall)
    assert tool_call.function.name == "record_value"
    assert tool_call.function.arguments["value"].lower() == "apple"

    final = model.complete(
        [
            {"role": "user", "content": "Record the value apple using the tool."},
            {
                "role": "assistant",
                "content": tool_response.content,
                "tool_calls": tool_response.tool_calls,
            },
            {
                "role": "tool",
                "content": "apple was recorded",
                "tool_call_id": tool_call.id,
            },
        ],
        tools=[RECORD_VALUE],
        inferenceConfig={"maxTokens": 50},
    )
    assert isinstance(final, str)
    assert final

    cache_attributes = get_bedrock_model_attributes(MODEL_ID)
    assert cache_attributes is not None
    assert cache_attributes.supports_explicit_caching
    cache_model = _model(MODEL_ID, cache="static")
    stable_prompt = "Reference material: " + ("The cached value is blue. " * 1_500)
    cache_messages = [
        {
            "role": "user",
            "content": stable_prompt + "\nReply with exactly: cached",
        }
    ]
    first = cache_model.complete(
        cache_messages,
        inferenceConfig={"maxTokens": 10},
    )
    assert cache_model.usage_metadata is not None
    assert cache_model.usage_metadata.cache_write_tokens > 0

    cache_messages.extend(
        [
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Reply with exactly: hit"},
        ]
    )
    cache_model.complete(
        cache_messages,
        inferenceConfig={"maxTokens": 10},
    )
    assert cache_model.usage_metadata is not None
    assert cache_model.usage_metadata.cache_read_tokens > 0

    reasoning_model = _model(
        REASONING_MODEL_ID,
        reasoning_effort="medium",
    )
    reasoning = reasoning_model.complete(
        [
            {
                "role": "user",
                "content": "Calculate 17 times 19, then give only the number.",
            }
        ],
        capture_thinking=True,
        inferenceConfig={"maxTokens": 2_000},
    )
    assert isinstance(reasoning, TokiThoughtResponse)
    assert "323" in reasoning.content
    assert reasoning.thought
    assert reasoning_model.last_provider_state is not None

    replayed = reasoning_model.complete(
        [
            {
                "role": "user",
                "content": "Calculate 17 times 19, then give only the number.",
            },
            {
                "role": "assistant",
                "content": reasoning.content,
                "provider_state": reasoning_model.last_provider_state,
            },
            {
                "role": "user",
                "content": "Now add one and give only the number.",
            },
        ],
        inferenceConfig={"maxTokens": 2_000},
    )
    assert "324" in replayed
