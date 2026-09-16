"""Unit tests for OpenAI Responses wire conversion. No network, no openai SDK."""

from toki.model import (
    TokiMessage,
    TokiToolCall,
    TokiToolFunction,
    _RawContentChunk,
    _RawProviderState,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawUsage,
)
from toki.openai.responses_wire import (
    PROVIDER_STATE_KEY,
    ResponsesStreamDecoder,
    flatten_tools,
    messages_to_input,
    turn_from_response,
    usage_from_wire,
)


def test_flatten_tools_chat_completions_shape():
    tools = flatten_tools([{
        "type": "function",
        "function": {
            "name": "record_value",
            "description": "Record a value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }])
    assert tools == [{
        "type": "function",
        "name": "record_value",
        "description": "Record a value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }]


def test_flatten_tools_none():
    assert flatten_tools(None) is None
    assert flatten_tools([]) is None


def test_messages_to_input_plain_chat():
    items = messages_to_input([
        TokiMessage(role="system", content="Be brief."),
        TokiMessage(role="user", content="Hi"),
        TokiMessage(role="assistant", content="Hello"),
    ])
    assert items == [
        {"type": "message", "role": "system", "content": "Be brief."},
        {"type": "message", "role": "user", "content": "Hi"},
        {"type": "message", "role": "assistant", "content": "Hello"},
    ]


def test_messages_to_input_replays_reasoning_then_function_call():
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "enc",
        "summary": [{"type": "summary_text", "text": "think"}],
    }
    assistant = TokiMessage(
        role="assistant",
        content="",
        tool_calls=[TokiToolCall(
            id="call_1",
            function=TokiToolFunction(name="record_value", arguments={"value": "apple"}),
        )],
        provider_state={PROVIDER_STATE_KEY: {"reasoning_items": [reasoning]}},
    )
    items = messages_to_input([
        TokiMessage(role="user", content="go"),
        assistant,
        TokiMessage(role="tool", content="ok", tool_call_id="call_1"),
    ])
    assert items[0] == {"type": "message", "role": "user", "content": "go"}
    assert items[1] == reasoning
    assert items[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "record_value",
        "arguments": '{"value": "apple"}',
    }
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }


def test_turn_from_response_extracts_text_tools_thought_and_state():
    response = {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "enc",
                "summary": [{"type": "summary_text", "text": "plan"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_9",
                "name": "record_value",
                "arguments": '{"value": "apple"}',
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    }
    turn = turn_from_response(response, capture_thinking=True)
    assert turn.content == "hi"
    assert turn.thought == "plan"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_9"
    assert turn.tool_calls[0].function.name == "record_value"
    assert turn.tool_calls[0].function.arguments == {"value": "apple"}
    assert turn.usage is not None
    assert turn.usage.prompt_tokens == 10
    assert turn.provider_state[PROVIDER_STATE_KEY]["reasoning_items"][0]["id"] == "rs_1"


def test_turn_from_response_omits_thought_when_not_capturing():
    response = {
        "output": [{
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "secret"}],
        }],
    }
    turn = turn_from_response(response, capture_thinking=False)
    assert turn.thought == ""
    assert turn.provider_state is not None


def test_usage_from_wire_cached_tokens():
    usage = usage_from_wire({
        "input_tokens": 100,
        "output_tokens": 5,
        "total_tokens": 105,
        "input_tokens_details": {"cached_tokens": 80},
    })
    assert usage is not None
    assert usage.cache_read_tokens == 80
    assert usage.total_tokens == 105


def test_stream_decoder_text_thought_tool_and_completed_state():
    decoder = ResponsesStreamDecoder(capture_thinking=True)
    chunks = [
        *decoder.feed({"type": "response.reasoning_summary_text.delta", "delta": "why"}),
        *decoder.feed({"type": "response.output_text.delta", "delta": "ok"}),
        *decoder.feed({
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "record_value",
                "arguments": "",
            },
        }),
        *decoder.feed({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 2,
            "delta": '{"value": "apple"}',
        }),
        *decoder.feed({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc",
                }],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        }),
    ]
    thoughts = [c for c in chunks if isinstance(c, _RawThoughtChunk)]
    texts = [c for c in chunks if isinstance(c, _RawContentChunk)]
    tools = [c for c in chunks if isinstance(c, _RawToolCallChunk)]
    usages = [c for c in chunks if isinstance(c, _RawUsage)]
    states = [c for c in chunks if isinstance(c, _RawProviderState)]
    assert [c.text for c in thoughts] == ["why"]
    assert [c.text for c in texts] == ["ok"]
    assert tools[0].id == "call_1" and tools[0].name == "record_value"
    assert tools[1].index == 2 and tools[1].arguments_fragment == '{"value": "apple"}'
    assert usages[0].usage.prompt_tokens == 3
    assert states[0].provider_state[PROVIDER_STATE_KEY]["reasoning_items"][0]["id"] == "rs_1"
