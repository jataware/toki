"""
Terminal rendering for the harness (Rich).

`Renderer.handle(event)` turns engine events into output: streamed markdown for
the reply, one-line cards for tools, diff panels for file changes. Nothing in
here touches the engine; it only draws.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment, Segments
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core import (
    Event,
    FileChanged,
    Mode,
    Notice,
    PermissionLevel,
    PermissionRequest,
    PlanUpdated,
    TextDelta,
    ThinkingDelta,
    ToolArgDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
)

MODE_COLORS = {Mode.AGENT: "green", Mode.ASK: "blue", Mode.PLAN: "magenta"}
DIFF_MAX_LINES = 80
TOOL_ICONS = {
    "list_dir": "▤", "read_file": "▢", "grep": "⌕", "find_files": "⌕",
    "edit_file": "✎", "create_file": "✚", "delete_file": "✕", "write_plan": "☰", "run_command": "❯",
}


def mode_color(mode: Mode) -> str:
    return MODE_COLORS[mode]


@dataclass
class _ToolLine:
    call_id: str
    name: str
    args: dict
    chars: int = 0
    current_arg: str = ""


@dataclass
class Renderer:
    console: Console
    show_thinking: bool = True
    _stream: "_MarkdownStream | None" = None
    _thinking_open: bool = False
    _tools: dict[str, _ToolLine] = field(default_factory=dict)
    _tool_live: Live | None = None

    # ----- lifecycle ---------------------------------------------------------

    def start_turn(self) -> None:
        self._tools.clear()

    def pause(self) -> None:
        """Stop any live region (before prompting the user for input)."""
        self._close_text()
        self._close_tool()

    def end_turn(self) -> None:
        self.pause()
        self.console.print()

    # ----- events ------------------------------------------------------------

    def handle(self, ev: Event) -> None:
        if isinstance(ev, ThinkingDelta):
            self._on_thinking(ev.text)
        elif isinstance(ev, TextDelta):
            self._on_text(ev.text)
        elif isinstance(ev, ToolStarted):
            self._on_tool_started(ev)
        elif isinstance(ev, ToolArgDelta):
            self._on_arg_delta(ev)
        elif isinstance(ev, ToolFinished):
            self._on_tool_finished(ev)
        elif isinstance(ev, FileChanged):
            self._on_file_changed(ev)
        elif isinstance(ev, PlanUpdated):
            self._on_plan(ev)
        elif isinstance(ev, Notice):
            self.notice(ev.text, ev.level)
        elif isinstance(ev, TurnEnded):
            self._close_text()
            self._close_tool()
            if ev.cancelled:
                self.console.print(Text("  ⏹ interrupted", style="yellow"))

    # ----- thinking ----------------------------------------------------------

    def _on_thinking(self, text: str) -> None:
        if not self.show_thinking:
            return
        self._close_tool()
        if not self._thinking_open:
            self.console.print(Text("  thinking", style="dim italic"))
            self._thinking_open = True
        self.console.print(Text(text, style="dim"), end="", soft_wrap=True)

    def _end_thinking(self) -> None:
        if self._thinking_open:
            self.console.print()
            self._thinking_open = False

    # ----- assistant text (streamed markdown) ---------------------------------

    def _on_text(self, text: str) -> None:
        self._end_thinking()
        self._close_tool()
        if self._stream is None:
            self._stream = _MarkdownStream(self.console)
        self._stream.push(text)

    def _close_text(self) -> None:
        self._end_thinking()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # ----- tools -------------------------------------------------------------

    def _tool_text(self, t: _ToolLine, *, done: bool = False, ok: bool = True, summary: str = "") -> Text:
        icon = TOOL_ICONS.get(t.name, "⚙")
        line = Text("  ")
        line.append(icon + " ", style="cyan" if not done else ("dim cyan" if ok else "red"))
        line.append(t.name, style="bold" if not done else "dim bold")
        if done:
            line.append("  " + summary, style="dim" if ok else "red")
        else:
            target = t.args.get("path") or t.args.get("command") or t.args.get("pattern") or t.args.get("glob") or t.args.get("title") or ""
            if target:
                line.append("  " + str(target), style="dim")
            if t.current_arg:
                line.append(f"  writing {t.current_arg} · {t.chars} chars", style="dim italic")
            else:
                line.append("  …", style="dim")
        return line

    def _on_tool_started(self, ev: ToolStarted) -> None:
        self._close_text()
        self._close_tool()
        t = _ToolLine(ev.call_id, ev.name, dict(ev.args))
        self._tools[ev.call_id] = t
        self._tool_live = Live(self._tool_text(t), console=self.console, refresh_per_second=12, transient=True)
        self._tool_live.start()

    def _on_arg_delta(self, ev: ToolArgDelta) -> None:
        t = self._tools.get(ev.call_id)
        if t is None:
            return
        if ev.arg in ("path", "command", "pattern", "glob", "title"):
            t.args[ev.arg] = t.args.get(ev.arg, "") + ev.text
            t.current_arg = ""
        else:
            if t.current_arg != ev.arg:
                t.current_arg, t.chars = ev.arg, 0
            t.chars += len(ev.text)
        if self._tool_live is not None:
            self._tool_live.update(self._tool_text(t))

    def _close_tool(self) -> None:
        if self._tool_live is not None:
            self._tool_live.stop()
            self._tool_live = None

    def _on_tool_finished(self, ev: ToolFinished) -> None:
        self._close_tool()
        t = self._tools.get(ev.call_id) or _ToolLine(ev.call_id, ev.name, ev.args)
        t.args = ev.args
        self.console.print(self._tool_text(t, done=True, ok=ev.ok, summary=ev.summary))
        if ev.name == "run_command" and ev.output:
            body = ev.output.split("\n", 1)[1] if ev.output.startswith("[") else ev.output
            body = body.strip()
            if body and body != "(no output)":
                lines = body.splitlines()
                shown = lines[:20]
                more = f"\n… {len(lines) - 20} more lines" if len(lines) > 20 else ""
                self.console.print(Text("\n".join("    " + l for l in shown) + more, style="dim"))
        elif not ev.ok:
            self.console.print(Text("    " + ev.output.splitlines()[0][:200], style="red dim"))

    # ----- file changes ------------------------------------------------------

    def _on_file_changed(self, ev: FileChanged) -> None:
        self._close_tool()
        self.diff_panel(ev.path, ev.diff, kind=ev.kind)

    def diff_panel(self, path: str, diff: str, *, kind: str = "modified", max_lines: int = DIFF_MAX_LINES) -> None:
        lines = diff.splitlines()
        # drop the ---/+++ header lines; the panel title carries the path
        body = [l for l in lines if not (l.startswith("---") or l.startswith("+++"))]
        extra = ""
        if len(body) > max_lines:
            extra = f"\n… {len(body) - max_lines} more lines (/diff {path} for all)"
            body = body[:max_lines]
        plus = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        minus = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        title = Text.assemble((f" {path} ", "bold"), (f"+{plus} ", "green"), (f"-{minus} ", "red"), (f"· {kind} ", "dim"))
        syntax = Syntax("\n".join(body) + extra, "diff", theme="ansi_dark", word_wrap=True, background_color="default")
        self.console.print(Panel(syntax, title=title, title_align="left", border_style="dim", padding=(0, 1)))

    def _on_plan(self, ev: PlanUpdated) -> None:
        self._close_tool()
        self.plan_panel(ev.markdown, ev.path)

    def plan_panel(self, markdown: str, path: str) -> None:
        self.console.print(Panel(Markdown(markdown), title=Text(f" plan · {path} ", style="bold magenta"), title_align="left", border_style="magenta", padding=(0, 1)))

    # ----- misc ---------------------------------------------------------------

    def notice(self, text: str, level: str = "info") -> None:
        self.pause()
        style = {"info": "dim", "warning": "yellow", "error": "bold red"}.get(level, "dim")
        self.console.print(Text("  " + text, style=style))

    def rule(self, title: str = "") -> None:
        self.console.print(Rule(title, style="dim"))

    def permission_prompt_text(self, req: PermissionRequest, *, offer_always: bool, always_prefix: str) -> Text:
        t = Text("\n  ⚡ ", style="yellow")
        t.append(req.tool, style="bold")
        t.append(" wants to run:\n")
        t.append("     " + req.detail + "\n", style="bold white")
        t.append("  [y] once   ", style="dim")
        if offer_always:
            t.append(f"[a] always ({always_prefix!r} → allowlist)   ", style="dim")
        t.append("[n] deny", style="dim")
        return t


class _MarkdownStream:
    """Stream markdown without polluting scrollback.

    A `Live` region redraws by moving the cursor up, which only works while the
    content fits on screen; past that, every refresh leaves another copy in the
    scrollback. So only the last screenful of the reply lives inside the Live
    region. Lines above it are printed once, permanently, as soon as they are far
    enough from the end that later text can no longer re-wrap them.
    """

    MIN_RENDER_INTERVAL = 1 / 15        # re-parsing markdown per token is wasteful on long replies

    def __init__(self, console: Console) -> None:
        self.console = console
        self.text = ""
        self.committed = 0                  # rendered lines already printed permanently
        self._last_render = 0.0
        self.live = Live(Text(""), console=console, refresh_per_second=12, transient=True, vertical_overflow="crop")
        self.live.start()

    def _tail(self) -> int:
        return max(4, self.console.size.height - 6)

    def _lines(self) -> list[list[Segment]]:
        return self.console.render_lines(Markdown(self.text), self.console.options, pad=False)

    def push(self, text: str) -> None:
        self.text += text
        now = time.monotonic()
        if now - self._last_render < self.MIN_RENDER_INTERVAL:
            return
        self._last_render = now
        lines = self._lines()
        stable = len(lines) - self._tail()
        if stable > self.committed:
            self._commit(lines[self.committed:stable])
            self.committed = stable
        self.live.update(_segments(lines[self.committed:]))

    def close(self) -> None:
        lines = self._lines()
        self.live.stop()                    # transient: erases the tail region
        self._commit(lines[self.committed:])
        self.committed = len(lines)

    def _commit(self, lines: list[list[Segment]]) -> None:
        if lines:
            self.console.print(_segments(lines), end="")


def _segments(lines: list[list[Segment]]) -> Segments:
    out: list[Segment] = []
    for line in lines:
        out.extend(line)
        out.append(Segment.line())
    return Segments(out)


# --- static screens -----------------------------------------------------------

def banner(console: Console, *, workspace: str, git: bool, model: str, mode: Mode, level: PermissionLevel, web: bool = True) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("workspace", Text(workspace) + Text("  git" if git else "  no git", style="dim"))
    grid.add_row("model", model)
    grid.add_row("mode", Text.assemble((mode.value, f"bold {mode_color(mode)}"), ("  · shell: " + level.value, "dim"), ("  · web: " + ("on" if web else "off"), "dim")))
    hint = Text("Type a message. ", style="dim")
    hint.append("@file", style="cyan")
    hint.append(" attaches context, ", style="dim")
    hint.append("/help", style="cyan")
    hint.append(" lists commands, ", style="dim")
    hint.append("/agent /ask /plan", style="cyan")
    hint.append(" switch modes.", style="dim")
    console.print(Panel(Group(grid, Text(""), hint), title=" toki harness ", title_align="left", border_style="dim"))
    if not git:
        console.print(Text("  tip: run `git init` first so edits show up as reviewable diffs in your editor.", style="yellow"))


def help_table(console: Console, commands: list, topic: str | None = None) -> None:
    if topic:
        cmd = next((c for c in commands if c.name == topic.lstrip("/") or topic.lstrip("/") in c.aliases), None)
        if cmd is None:
            console.print(Text(f"  no such command: {topic}", style="red"))
            return
        console.print(Text.assemble(("  /" + cmd.name, "bold cyan"), (" " + cmd.usage, "cyan"), ("\n  " + cmd.long or cmd.short, "")))
        return
    table = Table(box=None, padding=(0, 2), show_header=False, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    section = None
    for c in commands:
        if c.section != section:
            section = c.section
            table.add_row(Text(""), Text(""))
            table.add_row(Text(section, style="bold"), Text(""))
        table.add_row(f"/{c.name} {c.usage}".rstrip(), c.short)
    console.print(table)
    console.print()
    console.print(Text("  Modes:", style="bold"))
    for m in Mode:
        console.print(Text.assemble(("    " + m.value.ljust(7), f"bold {mode_color(m)}"), (m.blurb, "dim")))
    console.print(Text("  Shell permission levels:", style="bold"))
    for p in PermissionLevel:
        console.print(Text.assemble(("    " + p.value.ljust(10), "bold"), (p.blurb, "dim")))
    console.print(Text("\n  Mention files with @path to attach them. Alt+Enter inserts a newline. Ctrl+C stops a running turn.\n"
                       "  Conversations are saved to .harness/sessions/ after every turn; /sessions and /resume bring them back.", style="dim"))


def term_width() -> int:
    return shutil.get_terminal_size((100, 30)).columns
