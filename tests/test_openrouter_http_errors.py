"""Non-streaming OpenRouter calls check the HTTP status (no network).

JSON bodies carrying an `error` object keep raising the existing
`ValueError("Error from OpenRouter: ...")`, whatever the status; other non-2xx
responses raise the transport's status error with the status and a truncated
body; a 2xx non-JSON body raises a `ValueError` quoting it.
"""

import io
import json

import httpx
import pytest
import requests

import toki.openrouter.model as or_model
from toki import OpenRouterModel


MESSAGES = [{"role": "user", "content": "hi"}]
OK_BODY = json.dumps({
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
})
ERROR_BODY = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})
HTML_502 = "<html><body><h1>502 Bad Gateway</h1></body></html>"
REASONS = {200: "OK", 429: "Too Many Requests", 500: "Internal Server Error", 502: "Bad Gateway"}


@pytest.fixture
def reply(monkeypatch):
    """Configure the (status, body, headers) both transports answer with."""
    state: dict = {"status": 200, "body": OK_BODY, "headers": {}}

    def fake_post(self, url, **kwargs):
        r = requests.Response()
        r.status_code = state["status"]
        r.reason = REASONS.get(state["status"], "")
        r.headers.update(state["headers"])
        r.raw = io.BytesIO(state["body"].encode())
        r.encoding = "utf-8"
        r.url = url
        return r

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(state["status"], content=state["body"].encode(), headers=state["headers"])

    real_async_client = httpx.AsyncClient

    class MockAsyncClient(real_async_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    return state


async def _complete(asynchronous: bool):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    if asynchronous:
        return await model.acomplete(MESSAGES)
    return model.complete(MESSAGES)


def _status_error(asynchronous: bool) -> type[Exception]:
    return httpx.HTTPStatusError if asynchronous else requests.exceptions.HTTPError


both = pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])


@both
async def test_success_still_parses(reply, asynchronous):
    assert await _complete(asynchronous) == "hello"


@both
async def test_non_json_5xx_raises_status_error_with_body(reply, asynchronous):
    reply.update(status=502, body=HTML_502)
    with pytest.raises(_status_error(asynchronous), match=r"HTTP 502 Bad Gateway") as info:
        await _complete(asynchronous)
    assert "502 Bad Gateway</h1>" in str(info.value)
    assert info.value.response.status_code == 502


@both
async def test_long_error_body_is_truncated(reply, asynchronous):
    reply.update(status=500, body="x" * 5000)
    with pytest.raises(_status_error(asynchronous)) as info:
        await _complete(asynchronous)
    message = str(info.value)
    assert "[5000 chars total]" in message
    assert len(message) < or_model._ERROR_BODY_PREVIEW_CHARS + 200


@both
async def test_retry_after_is_reported_on_429(reply, asynchronous):
    reply.update(status=429, body="slow down", headers={"Retry-After": "7"})
    with pytest.raises(_status_error(asynchronous), match=r"HTTP 429 Too Many Requests \(Retry-After: 7\)") as info:
        await _complete(asynchronous)
    assert info.value.response.headers["Retry-After"] == "7"


@both
async def test_non_2xx_json_without_error_key_raises_status_error(reply, asynchronous):
    reply.update(status=500, body=json.dumps({"detail": "boom"}))
    with pytest.raises(_status_error(asynchronous), match=r"HTTP 500.*boom"):
        await _complete(asynchronous)


@both
@pytest.mark.parametrize("status", [200, 429, 502])
async def test_json_error_body_keeps_existing_value_error(reply, asynchronous, status):
    reply.update(status=status, body=ERROR_BODY)
    with pytest.raises(ValueError, match=r"Error from OpenRouter: .*Rate limit exceeded") as info:
        await _complete(asynchronous)
    assert not isinstance(info.value, (requests.exceptions.HTTPError, httpx.HTTPStatusError))


@both
async def test_2xx_non_json_raises_value_error_with_body(reply, asynchronous):
    reply.update(status=200, body="<html>maintenance</html>")
    with pytest.raises(ValueError, match=r"non-JSON response \(HTTP 200\).*maintenance"):
        await _complete(asynchronous)
