"""
The engine: modes, permissions, plan, change tracking, and the tool loop.

`Harness` owns a toki `Agent` and exposes everything a frontend needs as
methods and events. It never reads stdin or prints. The CLI in
`examples/harness/cli` is one thin consumer; a GUI would be another.

    harness = Harness(model, root=".", approver=my_prompt)
    for event in harness.run_turn("what does this project do?"):
        ...            # TextDelta, ToolStarted, ToolFinished, FileChanged, ...
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

from toki import (
    Agent,
    BaseModel,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolCallStream,
    TokiToolFunction,
    TokiUsageMetadata,
)

from .events import (
    Event,
    Notice,
    PlanUpdated,
    TextDelta,
    ThinkingDelta,
    ToolArgDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
)
from .permissions import Approver, PermissionLevel, Permissions, deny_all
from .sessions import SessionInfo, SessionStore
from .tools import PLAN_FILE, TOOLS, Tool, ToolContext, ToolResult
from .workspace import HARNESS_DIR, Workspace, WorkspaceError, truncate

CONFIG_FILE = "config.json"
MAX_TOOL_OUTPUT_CHARS = 20_000


class Mode(str, Enum):
    AGENT = "agent"
    ASK = "ask"
    PLAN = "plan"

    @property
    def blurb(self) -> str:
        return {
            Mode.AGENT: "reads, edits, and runs; edits show up as diffs in your editor",
            Mode.ASK: "read-only; explains and proposes but cannot change anything",
            Mode.PLAN: "researches, then writes a plan file for you to review; /build implements it",
        }[self]


@dataclass
class TurnRecord:
    """One user turn: where it starts in `agent.messages` and what it did to disk."""
    number: int
    start: int                                   # index of the user message
    text: str                                    # the user's message (attachments stripped)
    before: dict[str, str | None] = field(default_factory=dict)   # file content before this turn first touched it
    ran_shell: bool = False
    resumed: bool = False                        # loaded from a session file: no file checkpoints

    @property
    def files(self) -> list[str]:
        return sorted(self.before)


@dataclass
class UndoPreview:
    turns: list[TurnRecord]
    files: list[str]
    ran_shell: bool
    resumed: bool


@dataclass
class UndoResult:
    turns: list[TurnRecord]
    restored: list[str]          # files put back to their pre-turn content
    kept: list[str]              # files whose changes were kept
    ran_shell: bool


@dataclass
class _RedoEntry:
    messages: list[TokiMessage]
    turns: list[TurnRecord]
    after: dict[str, str | None]  # file content at undo time (only for files that were reverted)


class Harness:
    def __init__(
        self,
        model: BaseModel,
        root: str | Path = ".",
        *,
        approver: Approver = deny_all,
        mode: Mode = Mode.AGENT,
        permission_level: PermissionLevel | None = None,
        capture_thinking: bool = False,
        max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
    ) -> None:
        self.workspace = Workspace(Path(root))
        self.workspace.ensure_harness_dir()
        self.permissions = Permissions.load(self.config_path)
        if permission_level is not None:
            self.permissions.level = permission_level
        self.ctx = ToolContext(self.workspace, self.permissions, approver)
        self.capture_thinking = capture_thinking
        self.max_tool_output_chars = max_tool_output_chars
        self.mode = mode
        self.agent: Agent = Agent(model)
        self.usage = TokiUsageMetadata(0, 0, 0)
        self.turns = 0
        self._cancel = False
        self.sessions = SessionStore(self.workspace.harness_dir)
        self.session_path: Path | None = None      # where this conversation is saved (set on first save)
        self.autosave = True
        self.turn_records: list[TurnRecord] = []
        self._redo: list[_RedoEntry] = []
        self._refresh()

    # ----- configuration -----------------------------------------------------

    @property
    def model(self) -> BaseModel:
        return self.agent.model

    @property
    def config_path(self) -> Path:
        return self.workspace.harness_dir / CONFIG_FILE

    @property
    def plan_path(self) -> Path:
        return self.ctx.plan_path

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self._refresh()

    def set_permission_level(self, level: PermissionLevel) -> None:
        self.permissions.set_level(level)
        self._refresh()

    def allow(self, prefix: str) -> bool:
        return self.permissions.allow(prefix)

    def disallow(self, prefix: str) -> bool:
        return self.permissions.disallow(prefix)

    def set_web(self, enabled: bool) -> None:
        self.permissions.set_web(enabled)
        self._refresh()

    def tools_for(self, mode: Mode) -> list[Tool]:
        kinds = {
            Mode.ASK: {"read", "web"},
            Mode.PLAN: {"read", "plan", "web"},
            Mode.AGENT: {"read", "write", "plan", "shell", "web"},
        }[mode]
        if self.permissions.level is PermissionLevel.NONE:
            kinds = kinds - {"shell"}
        if not self.permissions.web:
            kinds = kinds - {"web"}
        return [t for t in TOOLS.values() if t.kind in kinds]

    @property
    def tools(self) -> list[Tool]:
        return self.tools_for(self.mode)

    def _refresh(self) -> None:
        """Rebind tools and the system prompt after a mode/permission change."""
        schemas = [t.schema() for t in self.tools]
        self.agent.tools = schemas or None
        prompt = self.system_prompt()
        if self.agent.messages and self.agent.messages[0].role == "system":
            self.agent.messages[0].content = prompt
        else:
            self.agent.messages.insert(0, _system(prompt))

    # ----- system prompt -----------------------------------------------------

    def system_prompt(self) -> str:
        ws = self.workspace
        lines = [
            f"You are a coding assistant working in the workspace at {ws.root}"
            + (" (a git repository)." if ws.is_git_repo() else "."),
            "Work only through the provided tools; paths are relative to the workspace root.",
            "Explore before you act: list_dir for layout, grep/find_files to locate things, read_file before editing. "
            "Never invent file contents.",
            "Be concise. Use markdown, refer to files by path, and finish with a short summary of what you did.",
            "",
        ]
        if self.mode is Mode.AGENT:
            lines += [
                "MODE: agent. You may change files with edit_file, create_file, and delete_file.",
                "Edits apply immediately; the user reviews them afterwards as a diff in their editor, so keep each edit "
                "focused and explain non-obvious changes. Make several small edit_file calls rather than rewriting a file.",
            ]
            if self.plan_path.exists():
                lines.append(
                    f"A plan exists at {PLAN_FILE_REL}. When implementing it, work through the steps in order and tick "
                    "each one off by editing its checkbox from '- [ ]' to '- [x]' as you complete it."
                )
        elif self.mode is Mode.ASK:
            lines += [
                "MODE: ask. You are read-only: you cannot edit files or run commands.",
                "Answer questions, explain code, and propose changes as snippets. If the user wants changes applied, "
                "tell them to switch with /agent.",
            ]
        else:
            lines += [
                "MODE: plan. Research the workspace, then call write_plan once with a concrete plan and stop.",
                "Do not implement anything. The plan should contain: a one-paragraph goal, '- [ ]' steps that name "
                "the files each step touches, and any risks or open questions. After writing it, summarize the plan "
                "in a few lines and tell the user they can review or edit it, then run /build to implement.",
            ]
        lines.append("")
        if any(t.kind == "shell" for t in self.tools):
            level = self.permissions.level
            lines.append(
                "SHELL: run_command is for tests, builds, git, and formatters only; use the file tools for reading and "
                "editing. "
                + {
                    PermissionLevel.ASK: "Every command asks the user for approval, so batch related commands.",
                    PermissionLevel.ALLOWLIST: "Some commands run automatically; others ask the user for approval.",
                    PermissionLevel.OPEN: "Commands run without asking; avoid destructive operations.",
                }.get(level, "")
            )
        else:
            lines.append("SHELL: none. If a command needs running, tell the user exactly what to run and why.")
        if any(t.kind == "web" for t in self.tools):
            lines.append(
                "WEB: web_search and fetch_url are available for documentation, library APIs, and error messages. "
                "Prefer the workspace first; search only when the answer is not local, and cite the URL you used."
            )
        return "\n".join(lines)

    # ----- conversation ------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh conversation (keeps mode, permissions, and change tracking)."""
        self.agent.messages = []
        self.turns = 0
        self.usage = TokiUsageMetadata(0, 0, 0)
        self.session_path = None
        self.turn_records = []
        self._redo.clear()
        self._refresh()

    # ----- undo / redo -------------------------------------------------------

    def history(self) -> list[TurnRecord]:
        return list(self.turn_records)

    def undo_preview(self, n: int = 1) -> UndoPreview | None:
        """What `undo(n)` would remove, so a frontend can ask about files first."""
        n = max(1, min(n, len(self.turn_records)))
        if not self.turn_records:
            return None
        turns = self.turn_records[-n:]
        files = sorted({f for t in turns for f in t.files})
        return UndoPreview(turns, files, any(t.ran_shell for t in turns), any(t.resumed for t in turns))

    def undo(self, n: int = 1, *, revert_files: bool = True) -> UndoResult | None:
        """Remove the last `n` turns from the conversation; optionally restore the files they changed."""
        preview = self.undo_preview(n)
        if preview is None:
            return None
        self._repair_history()
        turns = preview.turns
        start = turns[0].start
        removed = self.agent.messages[start:]
        del self.agent.messages[start:]
        del self.turn_records[-len(turns):]
        self.turns = max(0, self.turns - len(turns))
        restored: list[str] = []
        after: dict[str, str | None] = {}
        if revert_files:
            for t in reversed(turns):                       # newest first, so older checkpoints win
                for rel, content in t.before.items():
                    p = self.workspace.root / rel
                    if rel not in after:
                        after[rel] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
                    self._write_or_delete(p, content)
                    if rel not in restored:
                        restored.append(rel)
        kept = [f for f in preview.files if f not in restored]
        self._redo.append(_RedoEntry(removed, turns, after))
        self._autosave()
        return UndoResult(turns, restored, kept, preview.ran_shell)

    def redo(self) -> UndoResult | None:
        """Put back the most recently undone turn(s), including reverted file changes."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        self.agent.messages.extend(entry.messages)
        for t in entry.turns:
            t.start = t.start  # indices are unchanged: undo truncated exactly at t.start
        self.turn_records.extend(entry.turns)
        self.turns += len(entry.turns)
        for rel, content in entry.after.items():
            self._write_or_delete(self.workspace.root / rel, content)
        self._autosave()
        return UndoResult(entry.turns, sorted(entry.after), [], any(t.ran_shell for t in entry.turns))

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _write_or_delete(self, p: Path, content: str | None) -> None:
        rel = self.workspace.relpath(p)
        if content is None:
            if p.exists():
                self.workspace._remember(rel, p)
                p.unlink()
        else:
            self.workspace._remember(rel, p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    # ----- sessions ----------------------------------------------------------

    def list_sessions(self) -> list[SessionInfo]:
        return self.sessions.list()

    def save_session(self) -> Path | None:
        if len([m for m in self.agent.messages if m.role != "system"]) == 0:
            return None
        if self.session_path is None:
            self.session_path = self.sessions.new_path(self.agent.messages)
        self.sessions.save(self.session_path, messages=self.agent.messages, mode=self.mode.value, usage=self.usage, turns=self.turns)
        return self.session_path

    def resume_session(self, path: Path) -> list[TokiMessage]:
        """Load a saved conversation. Returns its messages (without the system prompt)."""
        data = self.sessions.load(path)
        self._repair_history()
        self.agent.messages = list(data["messages"])
        self.turns = int(data.get("turns", 0))
        self.usage = data["usage"]
        self.session_path = path
        self._redo.clear()
        self.turn_records = [
            TurnRecord(number=k + 1, start=i + 1, text=m.content.split("\n\n<attached", 1)[0].strip(), resumed=True)
            for k, (i, m) in enumerate((i, m) for i, m in enumerate(data["messages"]) if m.role == "user")
        ]   # +1: the system prompt is inserted at index 0 by _refresh()
        try:
            self.mode = Mode(data.get("mode", self.mode.value))
        except ValueError:
            pass
        self._refresh()
        self._repair_history()
        return list(data["messages"])

    def cancel(self) -> None:
        """Ask the running turn to stop after the current chunk."""
        self._cancel = True

    def run_turn(self, text: str, attachments: list[str] | None = None) -> Iterator[Event]:
        """Send a user message and drive the model → tools → model loop to a final answer."""
        self._cancel = False
        self._repair_history()
        self._redo.clear()
        record = TurnRecord(number=len(self.turn_records) + 1, start=len(self.agent.messages), text=text.strip())
        self.turn_records.append(record)
        self.agent.add_user_message(self._compose(text, attachments or []))
        self.turns += 1
        final_parts: list[str] = []
        try:
            while True:
                pending: list[TokiToolCall] = []
                text_parts: list[str] = []
                for chunk in self.agent.execute(stream=True, capture_thinking=self.capture_thinking):
                    if self._cancel:
                        raise _Cancelled(text_parts)
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                        yield TextDelta(chunk)
                    elif isinstance(chunk, TokiThinking):
                        yield ThinkingDelta(chunk.text)
                    elif isinstance(chunk, TokiToolCall):
                        pending.append(chunk)
                        yield ToolStarted(chunk.id, chunk.function.name, dict(chunk.function.arguments))
                    elif isinstance(chunk, TokiToolCallStream):
                        yield ToolStarted(chunk.id, chunk.name)
                        for arg_name, arg_stream in chunk.items():
                            for fragment in arg_stream:
                                if self._cancel:
                                    raise _Cancelled(text_parts)
                                yield ToolArgDelta(chunk.id, chunk.name, arg_name, fragment)
                        pending.append(TokiToolCall(chunk.id, TokiToolFunction(chunk.name, chunk.arguments), provider_state=chunk.provider_state))
                self._record_usage()
                if not pending:
                    final_parts.extend(text_parts)
                    break
                for call in pending:
                    if self._cancel:
                        self.agent.add_tool_message(call.id, "cancelled by user")
                        continue
                    self._checkpoint(record, call)
                    result = self._dispatch(call)
                    self.agent.add_tool_message(call.id, truncate(result.output, self.max_tool_output_chars))
                    yield ToolFinished(call.id, call.function.name, dict(call.function.arguments), result.output, result.ok, result.summary)
                    yield from result.events
                if self._cancel:
                    raise _Cancelled([])
        except _Cancelled as c:
            partial = "".join(c.text_parts)
            if partial:
                self.agent.add_assistant_message(partial + "\n\n[interrupted by user]")
            self._autosave()
            yield TurnEnded(partial, self.model.usage_metadata, cancelled=True)
            return
        self._autosave()
        yield TurnEnded("".join(final_parts), self.model.usage_metadata)

    def _autosave(self) -> None:
        if self.autosave:
            try:
                self.save_session()
            except OSError:
                pass

    def _checkpoint(self, record: TurnRecord, call: TokiToolCall) -> None:
        """Remember a file's content before this turn first changes it (for /undo)."""
        tool = TOOLS.get(call.function.name)
        if tool is None:
            return
        if tool.kind == "shell":
            record.ran_shell = True
            return
        if tool.kind not in ("write", "plan"):
            return
        args = call.function.arguments if isinstance(call.function.arguments, dict) else {}
        target = PLAN_FILE_REL if tool.kind == "plan" else args.get("path")
        if not target:
            return
        try:
            p = self.workspace.resolve(str(target))
        except WorkspaceError:
            return
        rel = self.workspace.relpath(p)
        if rel not in record.before:
            record.before[rel] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None

    def _dispatch(self, call: TokiToolCall) -> ToolResult:
        name = call.function.name
        tool = TOOLS.get(name)
        if tool is None or tool not in self.tools:
            return ToolResult(f"error: tool {name!r} is not available in {self.mode.value} mode", f"{name}: unavailable", ok=False)
        args = call.function.arguments if isinstance(call.function.arguments, dict) else {}
        return tool(self.ctx, **args)

    def _record_usage(self) -> None:
        u = self.model.usage_metadata
        if u is None:
            return
        self.usage = TokiUsageMetadata(
            prompt_tokens=self.usage.prompt_tokens + u.prompt_tokens,
            completion_tokens=self.usage.completion_tokens + u.completion_tokens,
            total_tokens=self.usage.total_tokens + u.total_tokens,
            cache_read_tokens=self.usage.cache_read_tokens + u.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens + u.cache_write_tokens,
        )

    def _repair_history(self) -> None:
        """Answer any tool calls left dangling by an interrupted turn."""
        msgs = self.agent.messages
        called: dict[str, None] = {}
        answered: set[str] = set()
        for m in msgs:
            if m.tool_calls:
                called.update({tc.id: None for tc in m.tool_calls})
            if m.role == "tool" and m.tool_call_id:
                answered.add(m.tool_call_id)
        for cid in called:
            if cid not in answered:
                self.agent.add_tool_message(cid, "cancelled by user")

    def abort(self) -> None:
        """Call after a KeyboardInterrupt escaped the generator: leaves history consistent."""
        self._cancel = True
        self._repair_history()
        self._autosave()

    def _compose(self, text: str, attachments: list[str]) -> str:
        parts = [text.strip()]
        for rel in attachments:
            try:
                p = self.workspace.resolve(rel)
                if p.is_dir():
                    parts.append(f"<attached directory {rel}>\n{self.workspace.tree(rel, depth=3)}\n</attached>")
                else:
                    body = truncate(self.workspace.read(rel), self.max_tool_output_chars)
                    parts.append(f"<attached file {rel}>\n{body}\n</attached>")
            except WorkspaceError as e:
                parts.append(f"<attachment {rel} unavailable: {e}>")
        return "\n\n".join(parts)

    # ----- plan --------------------------------------------------------------

    def plan_text(self) -> str | None:
        return self.plan_path.read_text(encoding="utf-8") if self.plan_path.exists() else None

    def write_plan(self, markdown: str) -> PlanUpdated:
        rel = self.workspace.relpath(self.plan_path)
        self.workspace.write(rel, markdown if markdown.endswith("\n") else markdown + "\n")
        return PlanUpdated(rel, markdown)

    def clear_plan(self) -> str | None:
        """Archive the active plan to .harness/plans/ and remove it. Returns the archive path."""
        if not self.plan_path.exists():
            return None
        text = self.plan_text() or ""
        title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("#")), "plan")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "plan"
        archive_dir = self.workspace.harness_dir / "plans"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = archive_dir / f"{stamp}-{slug}.md"
        dest.write_text(text, encoding="utf-8")
        self.plan_path.unlink()
        self.workspace._snapshots.pop(self.workspace.relpath(self.plan_path), None)
        self._refresh()
        return self.workspace.relpath(dest)

    def build_prompt(self) -> str | None:
        """Switch to agent mode and return the message that kicks off implementation."""
        plan = self.plan_text()
        if plan is None:
            return None
        self.set_mode(Mode.AGENT)
        return (
            f"Implement the plan in {PLAN_FILE_REL}. Work through the steps in order, editing the plan file to tick "
            "each step's checkbox as you finish it. Verify your work where possible and finish with a summary.\n\n"
            f"{plan}"
        )

    # ----- changes -----------------------------------------------------------

    def changes(self) -> list[tuple[str, str]]:
        """Changed workspace files this session (the plan and other .harness files are excluded)."""
        return [
            (rel, kind) for rel, kind in self.workspace.changed_files()
            if kind != "unchanged" and not rel.startswith(HARNESS_DIR + "/")
        ]

    def diff(self, rel: str) -> str:
        return self.workspace.diff(rel)

    def revert(self, rel: str) -> str:
        return self.workspace.revert(rel)

    def revert_all(self) -> list[str]:
        return [self.workspace.revert(rel) for rel, _ in self.changes()]

    def notice(self, text: str, level: str = "info") -> Notice:
        return Notice(text, level)  # type: ignore[arg-type]


PLAN_FILE_REL = f".harness/{PLAN_FILE}"
MENTION = re.compile(r"(?<!\S)@([\w./~-]+)")


def find_mentions(workspace: Workspace, text: str) -> list[str]:
    """`@path` tokens in `text` that resolve to something inside the workspace."""
    found: list[str] = []
    for m in MENTION.finditer(text):
        rel = m.group(1).rstrip(".,;:")
        try:
            if workspace.resolve(rel).exists() and rel not in found:
                found.append(rel)
        except WorkspaceError:
            continue
    return found


class _Cancelled(Exception):
    def __init__(self, text_parts: list[str]) -> None:
        self.text_parts = text_parts


def _system(content: str) -> TokiMessage:
    return TokiMessage(role="system", content=content)
