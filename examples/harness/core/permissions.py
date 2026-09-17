"""
Shell permission policy.

Four levels, independent of the agent/ask/plan mode:

- ``none``      the model has no shell tool at all
- ``ask``       every command needs approval
- ``allowlist`` commands matching a saved prefix run silently, the rest ask
- ``open``      nothing asks

File edits never ask: they are reviewed as a git diff (in Cursor or any editor)
after the fact. The `Approver` is the only place the core needs a human, and it
is injected so a GUI can supply a dialog instead of a terminal prompt.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal


class PermissionLevel(str, Enum):
    NONE = "none"
    ASK = "ask"
    ALLOWLIST = "allowlist"
    OPEN = "open"

    @property
    def blurb(self) -> str:
        return {
            PermissionLevel.NONE: "no shell access; file tools only",
            PermissionLevel.ASK: "every shell command asks for approval",
            PermissionLevel.ALLOWLIST: "allowlisted commands run; others ask",
            PermissionLevel.OPEN: "shell commands run without asking",
        }[self]


@dataclass
class PermissionRequest:
    """What the model wants to do. `detail` is the exact command."""
    tool: str
    detail: str


Decision = Literal["allow", "allow_always", "deny"]
Approver = Callable[[PermissionRequest], Decision]


def deny_all(_: PermissionRequest) -> Decision:
    """Default approver for headless use: never allow anything that asks."""
    return "deny"


@dataclass
class Permissions:
    level: PermissionLevel = PermissionLevel.ASK
    allowlist: list[str] = field(default_factory=list)
    web: bool = True           # may the model use web_search / fetch_url?
    path: Path | None = None   # when set, every change is saved here

    def check(self, command: str) -> Literal["allowed", "ask", "blocked"]:
        if self.level is PermissionLevel.NONE:
            return "blocked"
        if self.level is PermissionLevel.OPEN:
            return "allowed"
        if self.level is PermissionLevel.ALLOWLIST and self.matches(command):
            return "allowed"
        return "ask"

    def matches(self, command: str) -> bool:
        """True if every segment of a shell pipeline/chain matches a prefix."""
        for segment in _segments(command):
            words = _words(segment)
            if not any(_prefix_matches(_words(p), words) for p in self.allowlist):
                return False
        return True

    def allow(self, prefix: str) -> bool:
        prefix = prefix.strip()
        if not prefix or prefix in self.allowlist:
            return False
        self.allowlist.append(prefix)
        self.save()
        return True

    def disallow(self, prefix: str) -> bool:
        try:
            self.allowlist.remove(prefix.strip())
        except ValueError:
            return False
        self.save()
        return True

    def set_level(self, level: PermissionLevel) -> None:
        self.level = level
        self.save()

    def set_web(self, enabled: bool) -> None:
        self.web = enabled
        self.save()

    # ----- persistence -------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        path = path or self.path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"level": self.level.value, "allowlist": self.allowlist, "web": self.web}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, default_level: PermissionLevel = PermissionLevel.ASK) -> "Permissions":
        if not path.exists():
            return cls(level=default_level, path=path)
        try:
            data = json.loads(path.read_text())
            return cls(
                level=PermissionLevel(data.get("level", default_level.value)),
                allowlist=[str(p) for p in data.get("allowlist", [])],
                web=bool(data.get("web", True)),
                path=path,
            )
        except (ValueError, TypeError):
            return cls(level=default_level, path=path)


def _segments(command: str) -> list[str]:
    """Split `a && b | c ; d` into its component commands (naively, by operator)."""
    out, cur, i = [], [], 0
    quote = None
    while i < len(command):
        ch = command[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur.append(ch)
        elif command.startswith(("&&", "||"), i):
            out.append("".join(cur)); cur = []; i += 1
        elif ch in "|;":
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def _words(s: str) -> list[str]:
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


def _prefix_matches(prefix: list[str], words: list[str]) -> bool:
    return bool(prefix) and words[: len(prefix)] == prefix
