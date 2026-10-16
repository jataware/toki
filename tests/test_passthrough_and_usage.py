"""Offline tests: `Agent.execute` / `aexecute` forward extra kwargs to the model, and the OpenRouter
backend reports usage as a `TokiUsageMetadata` rather than the raw wire dict."""

import asyncio
import json

from toki import Agent, OpenRouterModel, TokiUsageMetadata
from toki.openrouter.model import _parse_sse_event, _turn_from_blocking_json


def _recording_model(calls: list[dict]) -> OpenRouterModel:
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="x")

    def complete(messages, *, stream=False, **kwargs):
        calls.append({"stream": stream, **kwargs})
        return iter(["a", "b"]) if stream else "ab"

    async def acomplete(messages, *, stream=False, **kwargs):
        calls.append({"stream": stream, **kwargs})
        if not stream:
            return "ab"

        async def gen():
            for chunk in ("a", "b"):
                yield chunk

        return gen()

    model.complete = complete  # type: ignore[method-assign]
    model.acomplete = acomplete  # type: ignore[method-assign]
    return model


def test_execute_forwards_kwargs():
    calls: list[dict] = []
    agent = Agent(_recording_model(calls))
    agent.add_user_message("hi")
    assert agent.execute(max_tokens=5, temperature=0) == "ab"
    assert "".join(agent.execute(stream=True, max_tokens=7)) == "ab"
    assert calls[0]["max_tokens"] == 5 and calls[0]["temperature"] == 0
    assert calls[1]["max_tokens"] == 7 and calls[1]["stream"] is True
    assert [m.content for m in agent.messages if m.role == "assistant"] == ["ab", "ab"]


def test_aexecute_forwards_kwargs():
    async def run(calls: list[dict]) -> None:
        model = _recording_model(calls)

        async def acomplete(messages, *, stream=False, **kwargs):
            calls.append({"stream": stream, **kwargs})
            if not stream:
                return "ab"

            async def gen():
                for chunk in ("a", "b"):
                    yield chunk

            return gen()

        model.acomplete = acomplete  # type: ignore[method-assign]
        agent = Agent(model)
        agent.add_user_message("hi")
        assert await agent.aexecute(max_tokens=3) == "ab"

    calls: list[dict] = []
    asyncio.run(run(calls))
    assert calls[0]["max_tokens"] == 3


def test_openrouter_blocking_usage_is_metadata():
    data = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1},
        },
    }
    usage = _turn_from_blocking_json(data, capture_thinking=False).usage
    assert usage == TokiUsageMetadata(
        prompt_tokens=10, completion_tokens=2, total_tokens=12, cache_read_tokens=4, cache_write_tokens=1
    )


def test_openrouter_stream_usage_is_metadata_and_null_usage_is_skipped():
    final = json.dumps({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    chunks = list(_parse_sse_event(final))
    assert chunks[-1].usage == TokiUsageMetadata(prompt_tokens=7, completion_tokens=3, total_tokens=10)

    middle = json.dumps({"choices": [{"delta": {"content": "x"}}], "usage": None})
    assert all(not hasattr(c, "usage") for c in _parse_sse_event(middle))
