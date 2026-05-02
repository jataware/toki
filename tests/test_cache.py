"""Unit tests for the conversation-caching machinery.

Covers `_CacheState`, the Anthropic `apply_cache_markers` helper, the
`_GoogleCacheManager` state machine (with a stubbed google-genai client), and
`OpenRouterModel._build_payload` cache dispatch. None of these tests touch a
network — caching behavior is fully exercised through in-process objects.
"""

import time
import warnings

import pytest

from toki import TokiMessage, OpenRouterModel
from toki.anthropic.utils import apply_cache_markers, build_cache_control
from toki.helpers.cache_state import _CacheState, _hash_messages, estimate_messages_tokens


# ---------- helpers ---------------------------------------------------------


def _msg(role: str, content: str) -> TokiMessage:
    return TokiMessage(role=role, content=content)


def _grow(messages: list[TokiMessage], n_chars: int) -> list[TokiMessage]:
    """Append a user message of `n_chars` length and a short assistant reply."""
    return [
        *messages,
        _msg("user", "x" * n_chars),
        _msg("assistant", "ok"),
    ]


def _big_user(n_chars: int) -> TokiMessage:
    return _msg("user", "x" * n_chars)


# ---------- _CacheState -----------------------------------------------------


class TestCacheState:
    def test_defers_when_below_min(self):
        state = _CacheState(min_cache_size_estimate=1024)
        msgs = [_msg("user", "tiny")]
        out = state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=estimate_messages_tokens(None, None, msgs),
        )
        assert out is None
        assert state.entries == []

    def test_anchors_on_first_sufficiently_large_call(self):
        state = _CacheState(min_cache_size_estimate=100)
        msgs = [_big_user(1000)]
        out = state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=estimate_messages_tokens(None, None, msgs),
        )
        assert out is not None
        assert out.anchor_index == 1
        assert len(state.entries) == 1

    def test_static_silent_reuse_when_prefix_unchanged(self):
        state = _CacheState(min_cache_size_estimate=100)
        msgs = [_big_user(1000)]
        first = state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=999,
        )
        bigger = [*msgs, _msg("assistant", "reply"), _msg("user", "follow-up")]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second = state.match_or_snapshot(
                strategy='static', messages=bigger, system=None, tools=None,
                prefix_token_estimate=999,
            )
        assert second is first

    def test_static_warns_and_snapshots_on_history_mutation(self):
        state = _CacheState(min_cache_size_estimate=100)
        original = [_big_user(1000)]
        state.match_or_snapshot(
            strategy='static', messages=original, system=None, tools=None,
            prefix_token_estimate=999,
        )
        # mutate the prefix message - hash no longer matches
        mutated = [_big_user(1500), _msg("assistant", "reply"), _msg("user", "u2")]
        with pytest.warns(UserWarning, match="history likely mutated"):
            state.match_or_snapshot(
                strategy='static', messages=mutated, system=None, tools=None,
                prefix_token_estimate=999,
            )
        assert len(state.entries) == 2

    def test_revert_silently_rehydrates_prior_entry(self):
        state = _CacheState(min_cache_size_estimate=100)
        original = [_big_user(1000)]
        first = state.match_or_snapshot(
            strategy='static', messages=original, system=None, tools=None,
            prefix_token_estimate=999,
        )
        mutated = [_big_user(1500)]
        with pytest.warns(UserWarning):
            state.match_or_snapshot(
                strategy='static', messages=mutated, system=None, tools=None,
                prefix_token_estimate=999,
            )
        # revert: original prefix again - should silently match the first entry
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reverted = state.match_or_snapshot(
                strategy='static', messages=original, system=None, tools=None,
                prefix_token_estimate=999,
            )
        assert reverted is first

    def test_clear(self):
        state = _CacheState(min_cache_size_estimate=100)
        msgs = [_big_user(1000)]
        state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=999,
        )
        assert state.entries
        state.clear()
        assert state.entries == []

    def test_cap_at_16_entries(self):
        state = _CacheState(min_cache_size_estimate=1)
        for i in range(20):
            msgs = [_msg("user", f"distinct-{i}-{'x' * 50}")]
            state.match_or_snapshot(
                strategy='static', messages=msgs, system=None, tools=None,
                prefix_token_estimate=999,
            )
        assert len(state.entries) == 16

    def test_lazy_prune_expired_entries(self):
        state = _CacheState(min_cache_size_estimate=100)
        msgs = [_big_user(1000)]
        entry = state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=999,
        )
        assert entry is not None
        entry.expires_at = time.time() - 1
        state.match_or_snapshot(
            strategy='static', messages=[_big_user(2000)], system=None, tools=None,
            prefix_token_estimate=999,
        )
        # the expired entry should have been pruned before the new snapshot
        assert all(e.anchor_index == 1 and len(state.entries) <= 2 for e in state.entries)
        assert entry not in state.entries

    def test_rolling_grows_with_refresh_delta(self):
        state = _CacheState(min_cache_size_estimate=10)
        msgs = [_msg("user", "hi"), _msg("assistant", "ok"), _big_user(400)]
        first = state.match_or_snapshot(
            strategy='rolling', messages=msgs, system=None, tools=None,
            prefix_token_estimate=100, refresh_delta_tokens=200,
        )
        bigger = [*msgs, _msg("assistant", "ok"), _big_user(8000)]
        second = state.match_or_snapshot(
            strategy='rolling', messages=bigger, system=None, tools=None,
            prefix_token_estimate=2000, refresh_delta_tokens=200,
        )
        assert first is not None and second is not None
        assert first is not second
        assert second.anchor_index > first.anchor_index


# ---------- mid-session strategy switching ---------------------------------


class TestStrategySwitching:
    def test_static_to_rolling_to_static_silent_reuse(self):
        state = _CacheState(min_cache_size_estimate=100)
        msgs = [_big_user(1000)]
        static_entry = state.match_or_snapshot(
            strategy='static', messages=msgs, system=None, tools=None,
            prefix_token_estimate=999,
        )
        bigger = [*msgs, _msg("assistant", "ok"), _big_user(800)]
        rolling_entry = state.match_or_snapshot(
            strategy='rolling', messages=bigger, system=None, tools=None,
            prefix_token_estimate=1500, refresh_delta_tokens=200,
        )
        assert rolling_entry is not None and rolling_entry is not static_entry
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            back = state.match_or_snapshot(
                strategy='static', messages=msgs, system=None, tools=None,
                prefix_token_estimate=999,
            )
        assert back is static_entry


# ---------- apply_cache_markers --------------------------------------------


class TestAnthropicMarkers:
    def _setup(self):
        wire_messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first-reply"},
            {"role": "user", "content": "second"},
        ]
        wire_tools = [
            {"type": "function", "function": {"name": "a", "description": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "description": "b", "parameters": {}}},
        ]
        return wire_messages, wire_tools

    def test_rolling_marks_system_last_tool_and_latest(self):
        wire_messages, wire_tools = self._setup()
        new_msgs, new_tools = apply_cache_markers(
            wire_messages, wire_tools, mode='rolling', anchor_index=0, ttl='5m',
        )
        assert isinstance(new_msgs[0]["content"], list)
        assert new_msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert new_tools[-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in new_tools[0]
        assert isinstance(new_msgs[-1]["content"], list)
        assert new_msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # untouched messages still pass by reference (string content preserved)
        assert new_msgs[1] is wire_messages[1]
        assert new_msgs[2] is wire_messages[2]

    def test_static_marks_anchor_message(self):
        wire_messages, wire_tools = self._setup()
        new_msgs, _ = apply_cache_markers(
            wire_messages, wire_tools, mode='static', anchor_index=2, ttl='5m',
        )
        # boundary = anchor_index - 1 = 1 (first user message)
        assert isinstance(new_msgs[1]["content"], list)
        assert new_msgs[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # latest message untouched
        assert new_msgs[-1] is wire_messages[-1]

    def test_ttl_1h(self):
        wire_messages, wire_tools = self._setup()
        new_msgs, new_tools = apply_cache_markers(
            wire_messages, wire_tools, mode='rolling', anchor_index=0, ttl='1h',
        )
        assert new_msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert new_tools[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_input_lists_untouched(self):
        wire_messages, wire_tools = self._setup()
        original_messages = [dict(m) for m in wire_messages]
        original_tools = [dict(t) for t in wire_tools]
        apply_cache_markers(wire_messages, wire_tools, mode='rolling', anchor_index=0)
        for orig, current in zip(original_messages, wire_messages):
            assert orig == current
        for orig, current in zip(original_tools, wire_tools):
            assert orig == current

    def test_build_cache_control_5m_default(self):
        assert build_cache_control('5m') == {"type": "ephemeral"}
        assert build_cache_control('1h') == {"type": "ephemeral", "ttl": "1h"}


# ---------- _GoogleCacheManager --------------------------------------------


class _FakeCache:
    def __init__(self, name: str, ttl_seconds: int = 3600):
        self.name = name
        self._expires_at = time.time() + ttl_seconds

    @property
    def expire_time(self):
        class _E:
            def __init__(self, t):
                self._t = t
            def timestamp(self):
                return self._t
        return _E(self._expires_at)


class _FakeClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.create_calls: list[dict] = []
        self.aio = self  # so `client.aio.caches.create` returns same object

    @property
    def caches(self):
        return self

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("simulated 'too small' rejection")
        return _FakeCache(name=f"cachedContents/fake-{len(self.create_calls)}")


def _make_google_manager(fake: _FakeClient, *, min_tokens: int = 100, refresh_delta: int = 200, ttl: int = 3600):
    from toki.google.model import _GoogleCacheManager
    mgr = _GoogleCacheManager(
        sdk_model_name="models/gemini-test",
        api_key="x",
        ttl_seconds=ttl,
        min_tokens=min_tokens,
        refresh_delta_tokens=refresh_delta,
        refresh_buffer_seconds=60,
    )
    mgr._client = fake
    return mgr


class TestGoogleCacheManager:
    def test_static_creates_then_reuses(self):
        fake = _FakeClient()
        mgr = _make_google_manager(fake)
        msgs = [_big_user(2000)]
        tail1, name1 = mgr.prepare(strategy='static', messages=msgs, tools_wire=None)
        assert name1 == "cachedContents/fake-1"
        assert tail1 == []
        bigger = [*msgs, _msg("assistant", "ok"), _msg("user", "u2")]
        tail2, name2 = mgr.prepare(strategy='static', messages=bigger, tools_wire=None)
        assert name2 == name1
        assert len(tail2) == 2
        assert len(fake.create_calls) == 1

    def test_static_recreates_at_same_anchor_when_expired(self):
        fake = _FakeClient()
        mgr = _make_google_manager(fake)
        msgs = [_big_user(2000)]
        _, name1 = mgr.prepare(strategy='static', messages=msgs, tools_wire=None)
        # force expiry
        mgr.state.entries[0].expires_at = time.time() - 1
        # _CacheState lazy-pruning will drop the expired entry, so a fresh
        # snapshot is taken at the same anchor (first viable call).
        _, name2 = mgr.prepare(strategy='static', messages=msgs, tools_wire=None)
        assert name2 == "cachedContents/fake-2"
        assert len(fake.create_calls) == 2

    def test_rolling_refresh_on_growth(self):
        fake = _FakeClient()
        mgr = _make_google_manager(fake, min_tokens=10, refresh_delta=200)
        # rolling caches messages[:-1], so big content has to be in the prefix
        m1 = [_big_user(2000), _msg("assistant", "ok"), _msg("user", "u")]
        _, name1 = mgr.prepare(strategy='rolling', messages=m1, tools_wire=None)
        assert name1 == "cachedContents/fake-1"
        m2 = [*m1, _msg("assistant", "ok"), _big_user(4000), _msg("user", "u3")]
        _, name2 = mgr.prepare(strategy='rolling', messages=m2, tools_wire=None)
        assert name2 == "cachedContents/fake-2"
        assert len(fake.create_calls) == 2

    def test_create_failure_falls_back(self):
        fake = _FakeClient(fail=True)
        mgr = _make_google_manager(fake)
        msgs = [_big_user(2000)]
        with pytest.warns(UserWarning, match="cache creation failed"):
            tail, name = mgr.prepare(strategy='static', messages=msgs, tools_wire=None)
        assert tail is None and name is None

    def test_too_small_returns_none(self):
        fake = _FakeClient()
        mgr = _make_google_manager(fake, min_tokens=10_000)
        msgs = [_msg("user", "tiny")]
        tail, name = mgr.prepare(strategy='static', messages=msgs, tools_wire=None)
        assert tail is None and name is None
        assert fake.create_calls == []


# ---------- OpenRouterModel dispatch ---------------------------------------


class TestOpenRouterDispatch:
    def _build(self, model: str, *, cache=None, cache_ttl='5m', messages=None):
        m = OpenRouterModel(model, api_key="x", cache=cache, cache_ttl=cache_ttl)
        return m, m._build_payload(messages or [_big_user(2000)], None, capture_thinking=False, stream=False, kwargs={})

    def test_anthropic_rolling_top_level(self):
        _, payload = self._build("anthropic/claude-test", cache='rolling')
        assert payload["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_rolling_top_level_1h(self):
        _, payload = self._build("anthropic/claude-test", cache='rolling', cache_ttl='1h')
        assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_anthropic_static_per_block_markers(self):
        msgs = [_big_user(8000)]  # plenty over 1024-token estimate
        _, payload = self._build("anthropic/claude-test", cache='static', messages=msgs)
        # static with anchor_index = 1, boundary = 0 → marker on the user message
        wire = payload["messages"]
        assert isinstance(wire[0]["content"], list)
        assert wire[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in payload  # no top-level marker on static

    def test_google_rolling_marks_latest(self):
        msgs = [_big_user(8000)]
        _, payload = self._build("google/gemini-test", cache='rolling', messages=msgs)
        wire = payload["messages"]
        assert isinstance(wire[-1]["content"], list)
        assert wire[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_google_static_marks_anchor(self):
        msgs = [_big_user(8000)]
        m, payload = self._build("google/gemini-test", cache='static', messages=msgs)
        wire = payload["messages"]
        assert isinstance(wire[0]["content"], list)
        assert wire[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_unknown_route_warns_at_construction(self):
        with pytest.warns(UserWarning, match="OpenRouter caching has no effect"):
            OpenRouterModel("meta-llama/llama-3.1-8b-instruct", api_key="x", cache='rolling')

    def test_invalidate_cache_clears_anchors(self):
        msgs = [_big_user(8000)]
        m, _ = self._build("anthropic/claude-test", cache='static', messages=msgs)
        assert m._cache_state.entries
        m.invalidate_cache()
        assert m._cache_state.entries == []


# ---------- _hash_messages stability ---------------------------------------


def test_hash_messages_stable_across_runs():
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    h1 = _hash_messages(None, None, msgs)
    h2 = _hash_messages(None, None, msgs)
    assert h1 == h2


def test_hash_messages_changes_with_content():
    msgs1 = [_msg("user", "hello")]
    msgs2 = [_msg("user", "world")]
    assert _hash_messages(None, None, msgs1) != _hash_messages(None, None, msgs2)
