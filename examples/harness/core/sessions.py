"""
Conversation persistence: one JSON file per session under `.harness/sessions/`.

A session holds the messages (minus the system prompt, which is regenerated
from the current mode and permissions), the mode, token usage, and a title
taken from the first user message. The harness saves after every turn; the
frontend offers the list on startup and via `/sessions`.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from toki import TokiMessage, TokiUsageMetadata

SESSIONS_DIR = "sessions"
TITLE_CHARS = 60


@dataclass
class SessionInfo:
    path: Path
    title: str
    updated: _dt.datetime
    turns: int
    mode: str

    @property
    def age(self) -> str:
        delta = _dt.datetime.now() - self.updated
        s = int(delta.total_seconds())
        if s < 60:
            return "just now"
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"


def title_from(messages: list[TokiMessage]) -> str:
    first = next((m for m in messages if m.role == "user"), None)
    if first is None:
        return "untitled"
    text = first.content.split("\n\n<attached", 1)[0].strip().replace("\n", " ")
    return (text[: TITLE_CHARS - 1] + "…") if len(text) > TITLE_CHARS else text or "untitled"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "session"


def message_to_dict(m: TokiMessage) -> dict:
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments},
             **({"provider_state": tc.provider_state} if tc.provider_state else {})}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.provider_state:
        d["provider_state"] = m.provider_state
    return d


class SessionStore:
    def __init__(self, harness_dir: Path) -> None:
        self.dir = harness_dir / SESSIONS_DIR

    def list(self) -> list[SessionInfo]:
        if not self.dir.exists():
            return []
        out = []
        for p in self.dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(SessionInfo(
                    path=p,
                    title=data.get("title", p.stem),
                    updated=_dt.datetime.fromisoformat(data["updated"]),
                    turns=int(data.get("turns", 0)),
                    mode=data.get("mode", "agent"),
                ))
            except (ValueError, KeyError, TypeError):
                continue
        return sorted(out, key=lambda s: s.updated, reverse=True)

    def new_path(self, messages: list[TokiMessage]) -> Path:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.dir / f"{stamp}-{_slug(title_from(messages))}.json"

    def save(self, path: Path, *, messages: list[TokiMessage], mode: str, usage: TokiUsageMetadata, turns: int) -> None:
        body = [message_to_dict(m) for m in messages if m.role != "system"]
        if not body:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "title": title_from(messages),
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "turns": turns,
            "usage": asdict(usage),
            "messages": body,
        }, indent=1, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> dict:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["messages"] = [TokiMessage.from_dict(m) for m in data["messages"]]
        u = data.get("usage") or {}
        data["usage"] = TokiUsageMetadata(**u) if u else TokiUsageMetadata(0, 0, 0)
        return data

    def delete(self, path: Path) -> None:
        path.unlink(missing_ok=True)
