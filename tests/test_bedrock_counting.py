import json

from toki import TokiMessage, TokiToolCall, TokiToolFunction
from toki.bedrock import counting


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return b'{"input_tokens": 37}'


def test_mantle_count_preserves_reasoning_tools_and_api_key(monkeypatch):
    captured: dict = {}

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(counting, "urlopen", urlopen)
    assistant = TokiMessage(
        role="assistant",
        content="calling",
        tool_calls=[
            TokiToolCall(
                id="call-1",
                function=TokiToolFunction(
                    name="lookup",
                    arguments={"value": "x"},
                ),
            )
        ],
        provider_state={
            "bedrock": {
                "reasoning_content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "think",
                                "signature": "signed",
                            }
                        }
                    }
                ]
            }
        },
    )
    messages = [
        TokiMessage(role="system", content="system"),
        TokiMessage(role="user", content="question"),
        assistant,
        TokiMessage(role="tool", content="result", tool_call_id="call-1"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {"type": "object"},
            },
        }
    ]

    count = counting.count_claude_tokens(
        session=None,
        region_name="us-east-1",
        model_id="global.anthropic.claude-sonnet-4-6",
        api_key="key",
        messages=messages,
        tools=tools,
    )

    request = captured["request"]
    payload = json.loads(request.data)
    assert count == 37
    assert request.get_header("X-api-key") == "key"
    assert payload["model"] == "anthropic.claude-sonnet-4-6"
    assert payload["system"] == "system"
    assert payload["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "think",
        "signature": "signed",
    }
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
