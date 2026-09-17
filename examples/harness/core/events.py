"""
Events emitted by `Harness.run_turn()`.

The core never prints. It yields one of these dataclasses for everything a
frontend might want to show, so the same engine can drive a terminal, a web
socket, or a desktop UI. A frontend that only cares about text can ignore the
rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from toki import TokiUsageMetadata


@dataclass
class TextDelta:
    """A fragment of the assistant's reply."""
    text: str


@dataclass
class ThinkingDelta:
    """A fragment of the model's reasoning (only with `capture_thinking=True`)."""
    text: str


@dataclass
class ToolStarted:
    """The model began a tool call. For streaming tools `args` is empty until
    `ToolArgDelta`s have arrived; check `ToolFinished.args` for the final dict."""
    call_id: str
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolArgDelta:
    """A fragment of one argument of a streaming tool call (e.g. the replacement
    text of `edit_file` as the model writes it)."""
    call_id: str
    name: str
    arg: str
    text: str


@dataclass
class ToolFinished:
    """A tool call ran (or was blocked / denied). `output` is what the model
    sees; `summary` is a one-line human description."""
    call_id: str
    name: str
    args: dict
    output: str
    ok: bool
    summary: str


@dataclass
class FileChanged:
    """A workspace file was created, modified, or deleted by a tool."""
    path: str
    kind: Literal["created", "modified", "deleted"]
    diff: str


@dataclass
class PlanUpdated:
    """The plan file was written."""
    path: str
    markdown: str


@dataclass
class Notice:
    """Informational message from the engine (mode changes, blocked tools, ...)."""
    text: str
    level: Literal["info", "warning", "error"] = "info"


@dataclass
class TurnEnded:
    """The model produced a final answer (or the turn was cancelled)."""
    text: str
    usage: TokiUsageMetadata | None
    cancelled: bool = False


Event = (
    TextDelta
    | ThinkingDelta
    | ToolStarted
    | ToolArgDelta
    | ToolFinished
    | FileChanged
    | PlanUpdated
    | Notice
    | TurnEnded
)
