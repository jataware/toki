"""Network timeouts are always set, configurable, and surface as clear errors.

No real network: `requests` / `httpx` are stubbed, except for the loopback
tests at the bottom, which talk to a local socket that accepts and then goes
silent to prove a stalled server raises instead of hanging.
"""

import json
import socket
import threading
import time

import httpx
import pytest
import requests
import urllib3

import toki.openrouter.model as or_model
from toki import OpenRouterModel


MESSAGES = [{"role": "user", "content": "hi"}]
BLOCKING_BODY = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
SSE_BODY = (
    b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    b"data: [DONE]\n\n"
)


def _requests_response(body: bytes, status: int = 200) -> requests.Response:
    import io

    r = requests.Response()
    r.status_code = status
    r.raw = io.BytesIO(body)
    r.url = or_model._API_URL
    return r


@pytest.fixture
def session_post(monkeypatch):
    """Stub `requests.Session.post`; records (session, kwargs) per call."""
    calls: list[tuple[requests.Session, dict]] = []
    state: dict = {"exc": None}

    def fake_post(self, url, **kwargs):
        calls.append((self, kwargs))
        if state["exc"] is not None:
            raise state["exc"]
        if kwargs.get("stream"):
            return _requests_response(SSE_BODY)
        return _requests_response(json.dumps(BLOCKING_BODY).encode())

    monkeypatch.setattr(requests.Session, "post", fake_post)
    fake_post.calls = calls
    fake_post.state = state
    return fake_post


@pytest.fixture
def httpx_clients(monkeypatch):
    """Swap `httpx.Client` / `httpx.AsyncClient` for subclasses that record the
    `timeout` they were built with and answer from a `MockTransport`."""
    timeouts: list[httpx.Timeout] = []
    state: dict = {"exc": None}
    real_client, real_async_client = httpx.Client, httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if state["exc"] is not None:
            raise state["exc"](request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=SSE_BODY, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=BLOCKING_BODY)

    class RecordingClient(real_client):
        def __init__(self, *, timeout, **kwargs):
            timeouts.append(timeout)
            super().__init__(timeout=timeout, transport=httpx.MockTransport(handler), **kwargs)

    class RecordingAsyncClient(real_async_client):
        def __init__(self, *, timeout, **kwargs):
            timeouts.append(timeout)
            super().__init__(timeout=timeout, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    monkeypatch.setattr(httpx, "AsyncClient", RecordingAsyncClient)
    return timeouts, state


# ----- OpenRouter sync path (requests) -----------------------------------------


def test_openrouter_blocking_passes_default_timeouts(session_post):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    assert model.complete(MESSAGES) == "hello"
    (_, kwargs), = session_post.calls
    assert kwargs["timeout"] == (
        or_model._OPENROUTER_CONNECT_TIMEOUT_DEFAULT,
        or_model._OPENROUTER_READ_TIMEOUT_DEFAULT,
    )
    # generous read budget for slow non-streaming reasoning calls
    assert or_model._OPENROUTER_READ_TIMEOUT_DEFAULT >= 300


def test_openrouter_streaming_passes_default_timeouts(session_post):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    assert "".join(model.complete(MESSAGES, stream=True)) == "hello"
    (_, kwargs), = session_post.calls
    assert kwargs["stream"] is True
    assert kwargs["timeout"] == (
        or_model._OPENROUTER_CONNECT_TIMEOUT_DEFAULT,
        or_model._OPENROUTER_STREAM_READ_TIMEOUT_DEFAULT,
    )


def test_openrouter_sync_timeouts_are_configurable(session_post):
    model = OpenRouterModel(
        "openai/gpt-4o-mini", api_key="dummy",
        connect_timeout=3, read_timeout=1200, stream_read_timeout=90,
    )
    model.complete(MESSAGES)
    list(model.complete(MESSAGES, stream=True))
    # attributes stay mutable, like `cache`
    model.read_timeout = None
    model.complete(MESSAGES)
    assert [kw["timeout"] for _, kw in session_post.calls] == [(3, 1200), (3, 90), (3, None)]


def test_openrouter_reuses_one_session_per_instance(session_post):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    model.complete(MESSAGES)
    list(model.complete(MESSAGES, stream=True))
    model.complete(MESSAGES)
    sessions = {id(s) for s, _ in session_post.calls}
    assert len(sessions) == 1
    adapter = session_post.calls[0][0].get_adapter(or_model._API_URL)
    assert adapter._pool_maxsize == or_model._OPENROUTER_POOL_MAXSIZE

    other = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    other.complete(MESSAGES)
    assert session_post.calls[-1][0] is not session_post.calls[0][0]


def test_openrouter_recreates_session_after_fork(session_post, monkeypatch):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    model.complete(MESSAGES)
    real_pid = or_model.os.getpid()
    monkeypatch.setattr(or_model.os, "getpid", lambda: real_pid + 1)
    model.complete(MESSAGES)
    assert session_post.calls[0][0] is not session_post.calls[1][0]


@pytest.mark.parametrize("stream", [False, True])
def test_openrouter_connect_timeout_is_clear(session_post, stream):
    session_post.state["exc"] = requests.exceptions.ConnectTimeout("boom")
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", connect_timeout=7)
    with pytest.raises(requests.exceptions.ConnectTimeout, match=r"connect_timeout=7s") as info:
        result = model.complete(MESSAGES, stream=stream)
        if stream:
            list(result)
    assert "openai/gpt-4o-mini" in str(info.value)
    assert isinstance(info.value.__cause__, requests.exceptions.ConnectTimeout)


@pytest.mark.parametrize("stream, attr", [(False, "read_timeout"), (True, "stream_read_timeout")])
def test_openrouter_read_timeout_is_clear(session_post, stream, attr):
    session_post.state["exc"] = requests.exceptions.ReadTimeout("boom")
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", **{attr: 42})
    with pytest.raises(requests.exceptions.ReadTimeout, match=rf"{attr}=42s"):
        result = model.complete(MESSAGES, stream=stream)
        if stream:
            list(result)


def test_openrouter_unrelated_connection_errors_pass_through_untouched(session_post):
    err = requests.exceptions.ConnectionError("connection refused")
    session_post.state["exc"] = err
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    with pytest.raises(requests.exceptions.ConnectionError) as info:
        model.complete(MESSAGES)
    assert info.value is err


# ----- OpenRouter async path + token counting (httpx) ----------------------------


async def test_openrouter_async_blocking_uses_read_timeout(httpx_clients):
    timeouts, _ = httpx_clients
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", connect_timeout=4, read_timeout=900)
    assert await model.acomplete(MESSAGES) == "hello"
    (t,) = timeouts
    assert (t.connect, t.read) == (4, 900)


async def test_openrouter_async_blocking_default_read_timeout_is_generous(httpx_clients):
    timeouts, _ = httpx_clients
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    await model.acomplete(MESSAGES)
    (t,) = timeouts
    assert (t.connect, t.read) == (
        or_model._OPENROUTER_CONNECT_TIMEOUT_DEFAULT,
        or_model._OPENROUTER_READ_TIMEOUT_DEFAULT,
    )


async def test_openrouter_async_streaming_uses_stream_read_timeout(httpx_clients):
    timeouts, _ = httpx_clients
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", connect_timeout=4, stream_read_timeout=30)
    chunks = [c async for c in model.acomplete(MESSAGES, stream=True)]
    assert "".join(chunks) == "hello"
    (t,) = timeouts
    assert (t.connect, t.read) == (4, 30)


async def test_openrouter_count_tokens_online_uses_configured_timeouts(httpx_clients):
    timeouts, _ = httpx_clients
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", connect_timeout=5, read_timeout=111)
    assert model.count_tokens(MESSAGES) == 3
    assert await model.acount_tokens(MESSAGES) == 3
    assert [(t.connect, t.read) for t in timeouts] == [(5, 111), (5, 111)]


@pytest.mark.parametrize(
    "exc, attr",
    [(httpx.ConnectTimeout, "connect_timeout"), (httpx.ReadTimeout, "read_timeout")],
)
async def test_openrouter_async_timeouts_are_clear(httpx_clients, exc, attr):
    _, state = httpx_clients
    state["exc"] = lambda request: exc("", request=request)
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy")
    with pytest.raises(exc, match=attr) as info:
        await model.acomplete(MESSAGES)
    assert info.value.request.url == or_model._API_URL


# ----- loopback: a server that accepts and then goes silent ---------------------


@pytest.fixture
def silent_server(monkeypatch):
    """Local HTTP server that reads the request, optionally sends response
    headers, then never sends another byte. Points `_API_URL` at it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    stop = threading.Event()
    mode = {"send_headers": False}
    conns: list[socket.socket] = []

    def serve():
        srv.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conns.append(conn)
            conn.recv(65536)
            if mode["send_headers"]:
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(or_model, "_API_URL", f"http://127.0.0.1:{srv.getsockname()[1]}/api/v1/chat/completions")
    yield mode
    stop.set()
    thread.join()
    for c in conns:
        c.close()
    srv.close()


def test_openrouter_blocking_silent_server_raises_instead_of_hanging(silent_server):
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", read_timeout=0.3)
    start = time.monotonic()
    with pytest.raises(requests.exceptions.ReadTimeout, match=r"read_timeout=0.3s"):
        model.complete(MESSAGES)
    assert time.monotonic() - start < 5


def test_openrouter_stalled_stream_raises_instead_of_hanging(silent_server):
    silent_server["send_headers"] = True
    model = OpenRouterModel("openai/gpt-4o-mini", api_key="dummy", stream_read_timeout=0.3)
    start = time.monotonic()
    # requests reports a mid-body stall as ConnectionError(ReadTimeoutError)
    with pytest.raises(requests.exceptions.ConnectionError, match=r"stream_read_timeout=0.3s") as info:
        list(model.complete(MESSAGES, stream=True))
    assert isinstance(info.value.__cause__.args[0], urllib3.exceptions.ReadTimeoutError)
    assert time.monotonic() - start < 5


# ----- other backends / dev scripts -----------------------------------------------


def test_ollama_clients_get_connect_timeout(monkeypatch):
    pytest.importorskip("ollama")
    from toki.ollama import model as ollama_model

    monkeypatch.setattr(ollama_model.OllamaModel, "_ensure_pulled", lambda self: None)
    m = ollama_model.OllamaModel("llama3.1:8b")
    for client in (m._client, m._async_client):
        t = client._client.timeout
        assert (t.connect, t.read) == (ollama_model._OLLAMA_CONNECT_TIMEOUT_DEFAULT, None)

    m = ollama_model.OllamaModel("llama3.1:8b", host="http://example.invalid:11434", connect_timeout=2, read_timeout=300)
    t = m._client._client.timeout
    assert (t.connect, t.read) == (2, 300)
    assert str(m._client._client.base_url).startswith("http://example.invalid:11434")


def test_google_cache_client_has_timeout():
    pytest.importorskip("google.genai")
    from toki.google import model as google_model

    mgr = google_model._GoogleCacheManager(
        sdk_model_name="models/gemini-2.5-flash", api_key="dummy", ttl_seconds=60,
        min_tokens=1, refresh_delta_tokens=1, refresh_buffer_seconds=1,
    )
    client = mgr._ensure_client()
    assert client._api_client._http_options.timeout == google_model._GOOGLE_CACHE_CREATE_TIMEOUT * 1000


def test_openrouter_fetch_models_has_timeout(monkeypatch):
    from toki.openrouter import fetch_models

    seen: dict = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        r = _requests_response(json.dumps({"data": []}).encode())
        r._content = r.raw.read()
        return r

    monkeypatch.setattr(fetch_models.requests, "get", fake_get)
    fetch_models._get_openrouter_models("dummy-key-for-timeout-test")
    assert seen["timeout"] is not None
