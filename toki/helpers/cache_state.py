"""Internal helper backing the conversation-caching `'rolling'` / `'static'`
strategies.

`_CacheState` tracks one or more historical *anchor* entries — a (anchor_index,
prefix_hash) pair that records "we cached the first N messages of the
conversation, and the prefix-content hashed to this." On each call, the owning
backend asks `match_or_snapshot(strategy=..., ...)` for the live anchor and
gets either:

  - an existing entry whose `prefix_hash` still matches the current call's
    `messages[:entry.anchor_index]` (silent reuse), or
  - a freshly-snapshotted entry (with a `UserWarning` emitted on the static
    path when prior entries existed but none matched — i.e. history mutation),
    or
  - `None` to mean "skip caching this turn" (deferred snapshot when the
    candidate prefix is below `min_cache_size_estimate`).

Strategy is passed *per call* so the model can mutate its `cache` attribute
between calls without re-instantiating state. The list of entries is capped at
16 (oldest dropped when full); entries with `expires_at` in the past are lazily
pruned at the top of every call.

Native Google additionally writes the `cachedContents/<id>` resource name into
each entry's `cache_name` and the server-reported `expire_time` into
`expires_at`. Anthropic and OpenRouter leave both fields `None`.
"""

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

from ..model import TokiMessage


_MAX_ENTRIES = 16


@dataclass
class _AnchorEntry:
    anchor_index: int
    prefix_hash: str
    prefix_token_estimate: int = 0
    cache_name: str | None = None
    expires_at: float | None = None


def _hash_messages(system: str | None, tools: list[dict] | None, messages: list[TokiMessage]) -> str:
    """Stable content-hash of the cacheable prefix.

    Includes the system instruction, tool schemas, and each message's
    role/content/tool_calls/tool_call_id. Order matters; tool-call dicts are
    serialized with sorted keys so `dict` ordering doesn't perturb the hash.
    """
    h = hashlib.sha256()
    h.update((system or "").encode("utf-8"))
    h.update(b"\x1e")  # record separator
    h.update(json.dumps(tools or [], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(b"\x1e")
    for m in messages:
        h.update(m.role.encode("utf-8"))
        h.update(b"\x1f")  # field separator
        h.update((m.content or "").encode("utf-8"))
        h.update(b"\x1f")
        if m.tool_calls:
            tc_payload = [
                {"id": tc.id, "type": tc.type, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in m.tool_calls
            ]
            h.update(json.dumps(tc_payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x1f")
        h.update((m.tool_call_id or "").encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


@dataclass
class _CacheState:
    """Anchor history for a single model instance.

    Lifetime is the model's; the list of entries is appended to (and pruned
    from) over the course of many `complete()` calls. Strategy is passed
    per-call so the user is free to flip `model.cache` between `'rolling'`,
    `'static'`, and `None` mid-session.
    """

    min_cache_size_estimate: int
    entries: list[_AnchorEntry] = field(default_factory=list)

    def clear(self) -> None:
        """Drop all anchor entries. Used by `model.invalidate_cache()`."""
        self.entries.clear()

    def match_or_snapshot(
        self,
        *,
        strategy: Literal['rolling', 'static'],
        messages: list[TokiMessage],
        system: str | None,
        tools: list[dict] | None,
        prefix_token_estimate: int,
        refresh_delta_tokens: int = 0,
    ) -> _AnchorEntry | None:
        """Find a still-valid historical anchor or snapshot a new one.

        Returns `None` to mean "skip caching this turn." This happens for
        `'static'` when the would-be snapshot is below
        `min_cache_size_estimate` (deferred anchor) and for `'rolling'` when
        there's nothing to cache yet (e.g. only one message in the list).

        On `'static'` with no match but prior entries exist, emits a
        `UserWarning` about the history mutation; old entries are kept for
        potential revert.
        """
        self._prune_expired()
        candidate_anchor = len(messages) if strategy == 'static' else len(messages) - 1
        if candidate_anchor <= 0:
            return None

        # walk newest-first looking for an entry whose cached prefix is a prefix
        # of the current messages list and still hashes to its recorded value.
        for entry in reversed(self.entries):
            if entry.anchor_index > len(messages):
                continue
            if _hash_messages(system, tools, messages[:entry.anchor_index]) != entry.prefix_hash:
                continue
            if strategy == 'rolling' and refresh_delta_tokens > 0:
                growth = prefix_token_estimate - entry.prefix_token_estimate
                if growth >= refresh_delta_tokens:
                    break  # match found, but stale enough that we want a refresh
            return entry

        # no usable entry — snapshot a new one if we're past the size threshold
        if prefix_token_estimate < self.min_cache_size_estimate:
            return None

        if strategy == 'static' and self.entries:
            from ..model import TokiCacheWarning
            warnings.warn(
                "cache='static': existing anchor's prefix doesn't match current messages "
                "(history likely mutated); snapshotting a new anchor. Prior anchors are "
                "retained so reverting to a previous prefix will silently rehydrate it.",
                category=TokiCacheWarning,
                stacklevel=3,
            )

        new_entry = _AnchorEntry(
            anchor_index=candidate_anchor,
            prefix_hash=_hash_messages(system, tools, messages[:candidate_anchor]),
            prefix_token_estimate=prefix_token_estimate,
        )
        self.entries.append(new_entry)
        if len(self.entries) > _MAX_ENTRIES:
            self.entries.pop(0)
        return new_entry

    def _prune_expired(self) -> None:
        if not self.entries:
            return
        now = time.time()
        self.entries[:] = [e for e in self.entries if e.expires_at is None or e.expires_at > now]


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate (chars/4). Used to gate cache writes
    without forcing a server round-trip per call."""
    return len(text) // 4


def estimate_messages_tokens(system: str | None, tools: list[dict] | None, messages: list[TokiMessage]) -> int:
    """Apply the chars/4 heuristic across system, tools, and message contents."""
    total = estimate_tokens(system or "")
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False))
    for m in messages:
        total += estimate_tokens(m.content or "")
        if m.tool_calls:
            for tc in m.tool_calls:
                total += estimate_tokens(tc.function.name)
                total += estimate_tokens(json.dumps(tc.function.arguments, ensure_ascii=False))
    return total
