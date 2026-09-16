import json
import os
from base64 import b64encode
from urllib.request import Request, urlopen

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from ..model import TokiMessage
from .models import normalize_bedrock_model_id


def _reasoning_blocks(message: TokiMessage) -> list[dict]:
    if message.provider_state is None:
        return []
    state = message.provider_state.get("bedrock")
    if state is None:
        return []
    blocks: list[dict] = []
    for wrapped in state["reasoning_content"]:
        reasoning = wrapped["reasoningContent"]
        if "reasoningText" in reasoning:
            text = reasoning["reasoningText"]
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": text["text"],
                    "signature": text["signature"],
                }
            )
        else:
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": b64encode(reasoning["redactedContent"]).decode("ascii"),
                }
            )
    return blocks


def _messages_to_anthropic(
    messages: list[TokiMessage],
) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    wire: list[dict] = []
    pending_results: list[dict] | None = None
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        if message.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            if pending_results is None:
                pending_results = [result]
                wire.append({"role": "user", "content": pending_results})
            else:
                pending_results.append(result)
            continue

        pending_results = None
        if message.role == "user":
            wire.append({"role": "user", "content": message.content})
            continue

        content = _reasoning_blocks(message)
        if message.content:
            content.append({"type": "text", "text": message.content})
        for call in message.tool_calls or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.function.name,
                    "input": call.function.arguments,
                }
            )
        wire.append({"role": "assistant", "content": content})
    return "\n".join(system_parts) or None, wire


def _tools_to_anthropic(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    result: list[dict] = []
    for tool in tools:
        function = tool["function"]
        converted = {
            "name": function["name"],
            "input_schema": function["parameters"],
        }
        if "description" in function:
            converted["description"] = function["description"]
        result.append(converted)
    return result


def count_claude_tokens(
    *,
    session,
    region_name: str,
    model_id: str,
    api_key: str | None,
    messages: list[TokiMessage],
    tools: list[dict] | None,
) -> int:
    """Count Claude CRIS tokens through Bedrock Mantle's Messages endpoint."""
    system, wire_messages = _messages_to_anthropic(messages)
    payload: dict = {
        "model": normalize_bedrock_model_id(model_id),
        "messages": wire_messages,
    }
    if system is not None:
        payload["system"] = system
    wire_tools = _tools_to_anthropic(tools)
    if wire_tools is not None:
        payload["tools"] = wire_tools

    url = (
        f"https://bedrock-mantle.{region_name}.api.aws/"
        "anthropic/v1/messages/count_tokens"
    )
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if key is not None:
        headers["x-api-key"] = key
    else:
        credentials = session.get_credentials().get_frozen_credentials()
        aws_request = AWSRequest(
            method="POST",
            url=url,
            data=body,
            headers=headers,
        )
        SigV4Auth(
            credentials,
            "bedrock-mantle",
            region_name,
        ).add_auth(aws_request)
        headers = dict(aws_request.headers.items())

    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read())
    return result["input_tokens"]
