"""
The terminal frontend: a thin loop over `Harness.run_turn()` events.

Everything user-facing lives here or in `ui.py` / `commands.py`; the engine in
`..core` has no idea a terminal exists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.formatted_text import FormattedText
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from ..core import Decision, Harness, Mode, PermissionLevel, PermissionRequest, SessionInfo, describe, find_mentions
from ..core.tools import _prefix_of
from . import commands
from .input import make_session
from .ui import Renderer, banner, mode_color

@dataclass
class App:
    harness: Harness
    console: Console = field(default_factory=Console)
    running: bool = True
    prefill: str = ""                     # text placed in the input box for the next prompt (after /undo)
    renderer: Renderer = field(init=False)

    def __post_init__(self) -> None:
        self.renderer = Renderer(self.console, show_thinking=True)
        self.harness.ctx.approver = self.approve
        self.session, self.completer = make_session(self.harness.workspace, self.toolbar)

    # ----- output helpers ----------------------------------------------------

    def info(self, text: str) -> None:
        self.renderer.notice(text, "info")

    def error(self, text: str) -> None:
        self.renderer.notice(text, "error")

    def show_status(self) -> None:
        h = self.harness
        self.console.print(Text.assemble(
            ("  mode ", "dim"), (h.mode.value, f"bold {mode_color(h.mode)}"), ("  " + h.mode.blurb, "dim"),
        ))
        self.console.print(Text.assemble(
            ("  shell ", "dim"), (h.permissions.level.value, "bold"), ("  " + h.permissions.level.blurb, "dim"),
        ))
        self.console.print(Text.assemble(("  model ", "dim"), (describe(h.model), "bold")))

    def set_mode(self, mode: Mode, *, announce: bool = True) -> None:
        if mode is self.harness.mode:
            if announce:
                self.info(f"already in {mode.value} mode · {mode.blurb}")
            return
        self.harness.set_mode(mode)
        if announce:
            self.console.print(Text.assemble(("  ▸ ", mode_color(mode)), (mode.value + " mode", f"bold {mode_color(mode)}"), ("  " + mode.blurb, "dim")))

    def toolbar(self) -> FormattedText:
        h = self.harness
        n = len(h.changes())
        parts = [
            ("class:bottom-toolbar", " "),
            (f"fg:ansi{mode_color(h.mode)} bold", h.mode.value),
            ("class:bottom-toolbar", f"  │  shell: {h.permissions.level.value}"),
            ("class:bottom-toolbar", "  │  web off" if not h.permissions.web else ""),
            ("class:bottom-toolbar", f"  │  {n} file{'s' if n != 1 else ''} changed" if n else ""),
            ("class:bottom-toolbar", f"  │  {h.usage.total_tokens:,} tokens" if h.usage.total_tokens else ""),
            ("class:bottom-toolbar", "  │  "),
            ("class:bottom-toolbar.key", "/help"),
            ("class:bottom-toolbar", "  Alt+Enter newline  Ctrl+C stop"),
        ]
        return FormattedText(parts)

    # ----- sessions ----------------------------------------------------------

    def session_table(self, sessions: list[SessionInfo], limit: int = 10) -> None:
        table = Table(box=None, padding=(0, 2), show_header=False, pad_edge=False)
        table.add_column(style="cyan", justify="right")
        table.add_column()
        table.add_column(style="dim")
        for i, sess in enumerate(sessions[:limit], 1):
            current = "  ← current" if sess.path == self.harness.session_path else ""
            table.add_row(str(i), Text(sess.title, style="bold") + Text(current, style="green"),
                          f"{sess.age} · {sess.turns} turn{'s' if sess.turns != 1 else ''} · {sess.mode}")
        self.console.print(table)

    def startup_picker(self) -> None:
        """Offer recent conversations; Enter starts a new one."""
        sessions = self.harness.list_sessions()
        if not sessions:
            return
        self.console.print(Text("  Recent conversations in this workspace:", style="bold"))
        self.session_table(sessions, limit=5)
        try:
            answer = self.console.input(Text("  resume [1-5] or Enter for new › ", style="dim")).strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return
        if answer.isdigit() and 1 <= int(answer) <= min(5, len(sessions)):
            self.resume(sessions[int(answer) - 1])
        else:
            self.info("new conversation")

    def resume(self, sess: SessionInfo) -> None:
        try:
            messages = self.harness.resume_session(sess.path)
        except Exception as e:
            self.error(f"could not load {sess.path.name}: {e}")
            return
        self.console.print(Text.assemble(("  ▸ resumed ", "green"), (sess.title, "bold"), (f"  ({sess.turns} turns, {sess.age})", "dim")))
        self._replay(messages)
        self.set_mode(self.harness.mode, announce=False)

    def _replay(self, messages, last: int = 4) -> None:
        """Show the tail of a resumed conversation so the user has context."""
        shown = [m for m in messages if m.role in ("user", "assistant")]
        if len(shown) > last:
            self.console.print(Text(f"  … {len(shown) - last} earlier messages", style="dim"))
            shown = shown[-last:]
        for m in shown:
            if m.role == "user":
                text = m.content.split("\n\n<attached", 1)[0]
                self.console.print(Text.assemble(("  you ", "dim"), (text[:300] + ("…" if len(text) > 300 else ""), "italic")))
            elif m.tool_calls:
                names = ", ".join(tc.function.name for tc in m.tool_calls)
                self.console.print(Text(f"  ⚙ {names}", style="dim"))
                if m.content.strip():
                    self.console.print(Markdown(m.content))
            elif m.content.strip():
                self.console.print(Markdown(m.content))
        self.console.print()

    def ask_choice(self, prompt: str, choices: dict[str, str]) -> str | None:
        """Single-key choice; returns the chosen value or None on Ctrl-C / EOF."""
        self.renderer.pause()
        while True:
            try:
                answer = self.console.input(Text(prompt, style="dim")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return None
            if answer in choices:
                return choices[answer]
            for key, value in choices.items():
                if answer == value:
                    return value
            self.console.print(Text("  " + " / ".join(choices), style="dim"))

    # ----- permission prompt -------------------------------------------------

    def approve(self, req: PermissionRequest) -> Decision:
        self.renderer.pause()
        offer_always = self.harness.permissions.level is PermissionLevel.ALLOWLIST
        prefix = _prefix_of(req.detail)
        self.console.print(self.renderer.permission_prompt_text(req, offer_always=offer_always, always_prefix=prefix))
        while True:
            try:
                answer = self.console.input("  › ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return "deny"
            if answer in ("y", "yes", ""):
                return "allow"
            if answer in ("n", "no"):
                return "deny"
            if answer in ("a", "always") and offer_always:
                return "allow_always"
            self.console.print(Text("  y / n" + (" / a" if offer_always else ""), style="dim"))

    # ----- turns -------------------------------------------------------------

    def run_turn(self, text: str, *, echo: str | None = None) -> None:
        attachments = self._mentions(text)
        if echo:
            self.console.print(Text.assemble(("  you ", "dim"), (echo, "italic")))
        if attachments:
            self.info("attached: " + ", ".join(attachments))
        self.renderer.start_turn()
        try:
            for ev in self.harness.run_turn(text, attachments):
                self.renderer.handle(ev)
        except KeyboardInterrupt:
            self.harness.abort()
            self.renderer.pause()
            self.console.print(Text("  ⏹ interrupted", style="yellow"))
        except Exception as e:  # provider errors, etc. — keep the session alive
            self.harness.abort()
            self.renderer.pause()
            self.error(f"{type(e).__name__}: {e}")
        finally:
            self.renderer.end_turn()
            self.completer.refresh()

    def _mentions(self, text: str) -> list[str]:
        return find_mentions(self.harness.workspace, text)

    # ----- main loop ---------------------------------------------------------

    def prompt_fragments(self) -> FormattedText:
        mode = self.harness.mode
        return FormattedText([(f"class:prompt.{mode.value}", mode.value), ("class:prompt.arrow", " ❯ ")])

    def loop(self, *, resume: str | None = None) -> None:
        """`resume`: None → offer the picker, "new" → skip it, "last" or a number → resume without asking."""
        h = self.harness
        banner(
            self.console, workspace=str(h.workspace.root), git=h.workspace.is_git_repo(),
            model=describe(h.model), mode=h.mode, level=h.permissions.level, web=h.permissions.web,
        )
        if resume is None:
            self.startup_picker()
        elif resume != "new":
            sessions = h.list_sessions()
            idx = 0 if resume == "last" else int(resume) - 1
            if 0 <= idx < len(sessions):
                self.resume(sessions[idx])
            else:
                self.error("no such saved conversation; starting a new one")
        while self.running:
            try:
                default, self.prefill = self.prefill, ""
                if default:
                    self.info("your message is back in the input box: edit it and press Enter, or clear it")
                line = self.session.prompt(self.prompt_fragments, default=default)
            except KeyboardInterrupt:
                self.info("press Ctrl+D or type /quit to exit")
                continue
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if commands.dispatch(self, line):
                continue
            self.run_turn(line)
        self.console.print(Text("  bye", style="dim"))


def run(root: str | Path, model_spec: str, *, mode: Mode, permission_level: PermissionLevel | None, thinking: bool,
        resume: str | None = None) -> int:
    from ..core import make_model

    console = Console()
    _route_warnings(console)
    try:
        model = make_model(model_spec)
    except Exception as e:
        console.print(Text(f"could not load model {model_spec!r}: {e}", style="bold red"))
        console.print(Text("set the provider's API key, pass --model provider:name, or try --model demo", style="dim"))
        return 2
    harness = Harness(model, root, mode=mode, permission_level=permission_level, capture_thinking=thinking)
    App(harness, console).loop(resume=resume)
    return 0


def _route_warnings(console: Console) -> None:
    """Show library warnings as one dim line each instead of a traceback."""
    def show(message, category, filename, lineno, file=None, line=None):
        console.print(Text(f"  ⚠ {category.__name__}: {message}", style="dim yellow"), soft_wrap=True)
    warnings.showwarning = show
    warnings.simplefilter("once")
