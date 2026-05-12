"""Real-API caching integration tests.

These tests actually hit each provider with a `cache=`-enabled model, make two
back-to-back calls, and assert that the second call shows a cache hit. They're
opt-in via the `cache_integration` pytest marker (excluded from the default
`pytest` run by `pyproject.toml`):

    pytest -m cache_integration

Per-backend verification — and a note on what *actually* hits the cache vs.
what just engages the cache infrastructure:

  - **Anthropic native** and **OpenRouter `anthropic/*`** — Anthropic's
    cache is keyed per-breakpoint by the *exact prefix hash up to the marker
    position*. Anthropic does **not** silently fall back to longer-cached
    prefixes from prior turns when the current request's marker lands at a
    new content position. So `'rolling'` (which advances the user marker
    each turn) writes a fresh cache entry every turn but doesn't read prior
    turns' caches. `'static'` (which pins markers at the snapshot
    boundary) is the deterministic-cache-hit story for these routes.
  - **Google native** — toki manages cache *names* directly via
    `google-genai`'s `client.caches.create()`, reusing the same
    `cachedContents/<id>` across turns until growth (rolling) or expiry
    (both) forces a refresh. So both modes produce reads.
  - **OpenRouter `google/*`** — Gemini's lookup is more permissive than
    Anthropic's for the rolling case; both modes produce reads.

Assertions therefore differ by route:
  - For routes where rolling produces reads (Google native, OpenRouter
    google), call 2 must show `cached_tokens > 0`.
  - For routes where rolling only writes (Anthropic native, OpenRouter
    anthropic), the rolling test only asserts that caching is engaged
    (`cache_creation > 0` on call 2). The static test asserts the read.

Each test hard-fails (rather than silently skipping) when the relevant API key
isn't set, matching the existing `tests/conftest.py` philosophy.
"""

import importlib.util
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from toki import (
    AnthropicModel,
    GoogleModel,
    OpenRouterModel,
    TokiMessage,
    TokiThoughtResponse,
    TokiToolsResponse,
    TokiToolsThoughtResponse,
)


pytestmark = pytest.mark.cache_integration


# -- helpers ----------------------------------------------------------------
#
# Build a system prompt large enough to clear every backend's caching floor
# (Anthropic / OpenRouter ≥ 1024 tokens, Google ≥ 4096). One paragraph
# repeated 800x is comfortably ~16k tokens by the chars/4 estimate.
_BIG_PARAGRAPH = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. Mitochondria generate most of the cell's supply of "
    "adenosine triphosphate (ATP), used as a source of chemical energy. "
)
_BIG_SYSTEM = "You are a helpful assistant. " + (_BIG_PARAGRAPH * 800)


def _result_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (TokiThoughtResponse, TokiToolsResponse, TokiToolsThoughtResponse)):
        return result.content
    raise AssertionError(f"unexpected result type: {type(result).__name__}")


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        pytest.fail(f"{var} is not set; required to run {var.split('_')[0].lower()} cache integration test")
    return val


@contextmanager
def _capture_litellm():
    """Patch `litellm.completion` / `litellm.acompletion` to capture each
    response object. Yields a list that gets appended to in call order."""
    captured: list = []
    import litellm
    real_completion = litellm.completion
    real_acompletion = litellm.acompletion

    def wrapped_completion(*args, **kwargs):
        resp = real_completion(*args, **kwargs)
        captured.append(resp)
        return resp

    async def wrapped_acompletion(*args, **kwargs):
        resp = await real_acompletion(*args, **kwargs)
        captured.append(resp)
        return resp

    with patch.object(litellm, "completion", wrapped_completion), \
         patch.object(litellm, "acompletion", wrapped_acompletion):
        yield captured


@contextmanager
def _capture_openrouter():
    """Patch `requests.post` (used by `OpenRouterModel`) to capture the parsed
    JSON of each response. Yields a list of decoded JSON dicts."""
    captured: list[dict] = []
    import requests
    real_post = requests.post

    def wrapped_post(url, *args, **kwargs):
        resp = real_post(url, *args, **kwargs)
        try:
            captured.append(resp.json())
        except Exception:
            captured.append({})
        return resp

    with patch.object(requests, "post", wrapped_post):
        yield captured


def _attr_or_get(obj, name, default=None):
    """litellm usage objects are pydantic-ish but sometimes leak through as
    plain dicts; tolerate both."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cache_read(usage) -> int:
    """Extract a cache-read token count from a litellm response.usage."""
    if usage is None:
        return 0
    direct = _attr_or_get(usage, "cache_read_input_tokens", 0) or 0
    if direct:
        return int(direct)
    details = _attr_or_get(usage, "prompt_tokens_details", None)
    return int(_attr_or_get(details, "cached_tokens", 0) or 0)


def _cache_creation(usage) -> int:
    """Extract a cache-write token count from a litellm response.usage."""
    if usage is None:
        return 0
    direct = _attr_or_get(usage, "cache_creation_input_tokens", 0) or 0
    if direct:
        return int(direct)
    details = _attr_or_get(usage, "prompt_tokens_details", None)
    return int(_attr_or_get(details, "cache_creation_tokens", 0) or 0)


def _openrouter_cached_tokens(payload: dict) -> int:
    """Pull cache-hit info out of an OpenRouter response's `usage` field."""
    usage = (payload or {}).get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return int(cached)


def _openrouter_cache_writes(payload: dict) -> int:
    """OpenRouter exposes Anthropic-style cache writes as
    `prompt_tokens_details.cache_write_tokens`."""
    usage = (payload or {}).get("usage") or {}
    written = (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0) or 0
    return int(written)


# -- tests ------------------------------------------------------------------


def _assert_cache_engaged(usage, *, label: str) -> None:
    """Either a write or a read counts as evidence that our cache_control
    markers reached the API and Anthropic acted on them."""
    creation = _cache_creation(usage)
    read = _cache_read(usage)
    assert (creation > 0) or (read > 0), (
        f"expected caching to be engaged on {label} (cache_creation > 0 or "
        f"cache_read > 0); got creation={creation}, read={read}; raw usage={usage!r}"
    )


@pytest.mark.parametrize("strategy", ["rolling", "static"])
def test_anthropic_native_caching(strategy: str):
    api_key = _require_env("ANTHROPIC_API_KEY")
    model = AnthropicModel(
        "claude-haiku-4-5",
        api_key=api_key,
        cache=strategy,
    )

    with _capture_litellm() as responses:
        first_messages = [
            TokiMessage(role="system", content=_BIG_SYSTEM),
            TokiMessage(role="user", content="Reply with a single word: 'one'."),
        ]
        first = model.complete(first_messages)
        assert _result_text(first)

        second_messages = [
            *first_messages,
            TokiMessage(role="assistant", content=_result_text(first)),
            TokiMessage(role="user", content="Now reply with a single word: 'two'."),
        ]
        second = model.complete(second_messages)
        assert _result_text(second)

    assert len(responses) == 2, f"expected 2 raw responses, got {len(responses)}"
    first_usage = _attr_or_get(responses[0], "usage")
    second_usage = _attr_or_get(responses[1], "usage")

    # Both calls should engage caching (either write or read). Call 1's
    # creation may be 0 if the cache was already warm from a previous run —
    # in that case call 1 is a read, which is still proof the markers reach
    # Anthropic and that caching works.
    _assert_cache_engaged(first_usage, label="the first call")
    _assert_cache_engaged(second_usage, label="the second call")

    if strategy == "static":
        # static pins the same marker positions every turn → call 2's prefix
        # hash matches call 1's cached entry exactly → deterministic hit.
        cached = _cache_read(second_usage)
        assert cached > 0, (
            f"expected cache_read > 0 on the second static-mode call; got {cached}; "
            f"raw usage={second_usage!r}"
        )


@pytest.mark.parametrize("strategy", ["rolling", "static"])
def test_google_native_caching(strategy: str):
    api_key = _require_env("GEMINI_API_KEY")
    if importlib.util.find_spec("google.genai") is None:
        pytest.fail(
            "google-genai is not installed; native GoogleModel caching needs it. "
            "Install with `uv sync --extra google` (or `--extra all`)."
        )
    model = GoogleModel(
        "gemini-2.5-flash",
        api_key=api_key,
        cache=strategy,
        cache_min_tokens=1024,
    )

    # Synthesize a small priming turn so the rolling prefix on call 1 is not
    # system-only (Gemini's cachedContents.create() rejects empty contents,
    # and rolling caches `messages[:-1]`, so we need at least one
    # user/assistant pair before the latest user message).
    first_messages = [
        TokiMessage(role="system", content=_BIG_SYSTEM),
        TokiMessage(role="user", content="Hello!"),
        TokiMessage(role="assistant", content="Hi! How can I help?"),
        TokiMessage(role="user", content="Reply with a single word: 'one'."),
    ]
    with _capture_litellm() as responses:
        first = model.complete(first_messages)
        assert _result_text(first)

        second_messages = [
            *first_messages,
            TokiMessage(role="assistant", content=_result_text(first)),
            TokiMessage(role="user", content="Now reply with a single word: 'two'."),
        ]
        second = model.complete(second_messages)
        assert _result_text(second)

    entries = model._cache_manager.state.entries
    assert entries, "expected _CacheState to have at least one entry after the calls"
    first_entry = entries[0]
    assert first_entry.cache_name and first_entry.cache_name.startswith("cachedContents/"), (
        f"expected a cachedContents/<id> resource name, got {first_entry.cache_name!r}"
    )
    assert first_entry.expires_at is not None

    # Verify the second call actually read from the cache. For Gemini, litellm
    # surfaces this as `prompt_tokens_details.cached_tokens`.
    assert len(responses) == 2
    second_usage = _attr_or_get(responses[1], "usage")
    cached = _cache_read(second_usage)
    assert cached > 0, (
        f"expected cached_tokens > 0 on the second Gemini response; "
        f"raw usage={second_usage!r}"
    )

    if strategy == "static":
        # static should reuse the same anchor's cache_name on call 2.
        assert any(e.cache_name == first_entry.cache_name for e in model._cache_manager.state.entries), (
            "expected the original static anchor to still be live after the second call"
        )


@pytest.mark.parametrize("strategy", ["rolling", "static"])
def test_openrouter_anthropic_caching(strategy: str):
    api_key = _require_env("OPENROUTER_API_KEY")
    model = OpenRouterModel(
        "anthropic/claude-haiku-4.5",
        api_key=api_key,
        cache=strategy,
    )

    with _capture_openrouter() as responses:
        first_messages = [
            TokiMessage(role="system", content=_BIG_SYSTEM),
            TokiMessage(role="user", content="Reply with a single word: 'one'."),
        ]
        first = model.complete(first_messages)
        assert _result_text(first)

        second_messages = [
            *first_messages,
            TokiMessage(role="assistant", content=_result_text(first)),
            TokiMessage(role="user", content="Now reply with a single word: 'two'."),
        ]
        second = model.complete(second_messages)
        assert _result_text(second)

    assert len(responses) == 2
    second_cached = _openrouter_cached_tokens(responses[1])
    second_writes = _openrouter_cache_writes(responses[1])

    if strategy == "static":
        # Static pins the same marker positions every turn → call 2's prefix
        # hash matches call 1's cached entry exactly → deterministic hit.
        assert second_cached > 0, (
            f"expected cached_tokens > 0 on the second OpenRouter static response; "
            f"got usage={(responses[1] or {}).get('usage')!r}"
        )
    else:
        # Rolling on `anthropic/*` advances the user marker each turn, which
        # makes call 2 write a fresh entry rather than read call 1's
        # (Anthropic's per-breakpoint cache lookup is keyed on the exact
        # prefix hash up to the marker position). All we can verify here is
        # that caching infra is engaged on call 2.
        assert (second_writes > 0) or (second_cached > 0), (
            f"expected caching to be engaged on the second OpenRouter rolling "
            f"response (cache_write_tokens > 0 or cached_tokens > 0); got "
            f"usage={(responses[1] or {}).get('usage')!r}"
        )


@pytest.mark.parametrize("strategy", ["rolling", "static"])
def test_openrouter_google_caching(strategy: str):
    api_key = _require_env("OPENROUTER_API_KEY")
    model = OpenRouterModel(
        "google/gemini-2.5-flash",
        api_key=api_key,
        cache=strategy,
    )

    with _capture_openrouter() as responses:
        first_messages = [
            TokiMessage(role="system", content=_BIG_SYSTEM),
            TokiMessage(role="user", content="Reply with a single word: 'one'."),
        ]
        first = model.complete(first_messages)
        assert _result_text(first)

        second_messages = [
            *first_messages,
            TokiMessage(role="assistant", content=_result_text(first)),
            TokiMessage(role="user", content="Now reply with a single word: 'two'."),
        ]
        second = model.complete(second_messages)
        assert _result_text(second)

    assert len(responses) == 2
    cached = _openrouter_cached_tokens(responses[1])
    assert cached > 0, (
        f"expected cached_tokens > 0 on the second OpenRouter response; got usage="
        f"{(responses[1] or {}).get('usage')!r}"
    )
