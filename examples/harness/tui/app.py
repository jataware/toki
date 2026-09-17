"""
Textual frontend for the harness: a chat pane with per-message rewind buttons,
collapsible tool cards, diff cards with keep/revert, and a sidebar for mode,
permissions, changes, plan, and sessions.

    python -m examples.harness --tui [workspace]

Same engine as the plain CLI (`examples.harness.core`); nothing here touches
files or models directly. The engine runs in a worker thread and every event is
marshalled to the UI thread with `call_from_thread`, so the approver and the
undo prompt are modal dialogs that block the worker until answered.
"""

from __future__ import annotations

import threading
from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
    TextArea,
)

from ..core import (
    Decision,
    FileChanged,
    Harness,
    Mode,
    Notice,
    PermissionLevel,
    PermissionRequest,
    PlanUpdated,
    SessionInfo,
    TextDelta,
    ThinkingDelta,
    ToolArgDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    TurnRecord,
    UndoPreview,
    describe,
    find_mentions,
)
from toki import TokiMessage

TOOL_ICONS = {
    "list_dir": "▤", "read_file": "▢", "grep": "⌕", "find_files": "⌕", "edit_file": "✎", "create_file": "✚",
    "delete_file": "✕", "write_plan": "☰", "run_command": "❯", "web_search": "◎", "fetch_url": "⇩",
}
MODE_ORDER = [Mode.AGENT, Mode.ASK, Mode.PLAN]


# --- chat widgets --------------------------------------------------------------

class TurnWidget(Widget):
    """Anything in the chat that belongs to a turn (so /undo can remove it)."""
    def __init__(self, *args, turn: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn = turn


class UserMessage(TurnWidget):
    DEFAULT_CSS = """
    UserMessage { height: auto; margin: 1 0 0 0; padding: 0 1; border-left: thick $primary; background: $boost; }
    UserMessage Horizontal { height: auto; }
    UserMessage .who { color: $text-muted; width: 1fr; }
    UserMessage .rewind { min-width: 12; background: transparent; color: $text-muted; }
    UserMessage .rewind:hover { color: $warning; background: $surface; }
    UserMessage .body { padding: 0 0 0 0; }
    """

    class Rewind(Message):
        def __init__(self, turn: int) -> None:
            super().__init__()
            self.turn = turn

    def __init__(self, text: str, turn: int) -> None:
        super().__init__(turn=turn)
        self.text = text

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"you · turn {self.turn}", classes="who")
            yield Button("↶ rewind", classes="rewind", variant="default", compact=True)
        yield Static(self.text, classes="body")

    @on(Button.Pressed, ".rewind")
    def _rewind(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Rewind(self.turn))


class AssistantMessage(TurnWidget):
    DEFAULT_CSS = """
    AssistantMessage { height: auto; margin: 1 0 0 0; padding: 0 1; }
    AssistantMessage Markdown { margin: 0; padding: 0; background: transparent; }
    AssistantMessage .thinking { color: $text-muted; text-style: italic; }
    """

    def __init__(self, turn: int, *, text: str = "") -> None:
        super().__init__(turn=turn)
        self._initial = text
        self._stream = None
        self.thinking = ""

    def compose(self) -> ComposeResult:
        yield Static("", classes="thinking")
        yield Markdown(self._initial)

    async def append(self, text: str) -> None:
        md = self.query_one(Markdown)
        if self._stream is None:
            self._stream = Markdown.get_stream(md)
        await self._stream.write(text)

    async def append_thinking(self, text: str) -> None:
        self.thinking += text
        self.query_one(".thinking", Static).update(Text("thinking  " + self.thinking[-400:], style="italic"))

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class ToolCard(TurnWidget):
    DEFAULT_CSS = """
    ToolCard { height: auto; margin: 0 0 0 1; }
    ToolCard Collapsible { border: none; padding: 0; margin: 0; background: transparent; }
    ToolCard CollapsibleTitle { padding: 0 1; color: $text-muted; }
    ToolCard CollapsibleTitle:hover { background: $surface; color: $text; }
    ToolCard .output { color: $text-muted; padding: 0 2; }
    ToolCard.error CollapsibleTitle { color: $error; }
    """

    def __init__(self, call_id: str, name: str, turn: int, *, args: dict | None = None) -> None:
        super().__init__(turn=turn)
        self.call_id = call_id
        self.tool_name = name
        self.args = dict(args or {})
        self.chars = 0
        self.current_arg = ""

    def compose(self) -> ComposeResult:
        with Collapsible(title=self._title(), collapsed=True):
            yield Static("", classes="output")

    def _target(self) -> str:
        for key in ("path", "command", "pattern", "glob", "title", "query", "url"):
            if self.args.get(key):
                return str(self.args[key])
        return ""

    def _title(self, summary: str | None = None) -> str:
        icon = TOOL_ICONS.get(self.tool_name, "⚙")
        if summary is not None:
            return f"{icon} {self.tool_name}  {summary}"
        tail = f"  writing {self.current_arg} · {self.chars} chars" if self.current_arg else "  …"
        return f"{icon} {self.tool_name}  {self._target()}{tail}"

    def arg_delta(self, arg: str, text: str) -> None:
        if arg in ("path", "command", "pattern", "glob", "title", "query", "url"):
            self.args[arg] = self.args.get(arg, "") + text
            self.current_arg = ""
        else:
            if self.current_arg != arg:
                self.current_arg, self.chars = arg, 0
            self.chars += len(text)
        self.query_one(Collapsible).title = self._title()

    def finish(self, summary: str, output: str, ok: bool) -> None:
        self.query_one(Collapsible).title = self._title(summary)
        shown = output if len(output) <= 4000 else output[:4000] + f"\n… ({len(output) - 4000} more chars)"
        self.query_one(".output", Static).update(shown)
        if not ok:
            self.add_class("error")
            self.query_one(Collapsible).collapsed = False


class DiffCard(TurnWidget):
    DEFAULT_CSS = """
    DiffCard { height: auto; margin: 0 0 0 1; }
    DiffCard Collapsible { border: round $secondary 30%; padding: 0; margin: 0; background: transparent; }
    DiffCard CollapsibleTitle { padding: 0 1; }
    DiffCard .actions { height: auto; padding: 0 1; }
    DiffCard .actions Button { min-width: 10; margin: 0 1 0 0; }
    DiffCard.reverted CollapsibleTitle { color: $text-muted; text-style: strike; }
    """

    class Revert(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, path: str, kind: str, diff: str, turn: int) -> None:
        super().__init__(turn=turn)
        self.path = path
        self.kind = kind
        self.diff = diff

    def compose(self) -> ComposeResult:
        lines = [l for l in self.diff.splitlines() if not l.startswith(("---", "+++"))]
        plus = sum(1 for l in lines if l.startswith("+"))
        minus = sum(1 for l in lines if l.startswith("-"))
        title = f"{self.path}  +{plus} -{minus} · {self.kind}"
        with Collapsible(title=title, collapsed=len(lines) > 40):
            yield Static(Syntax("\n".join(lines), "diff", theme="ansi_dark", background_color="default", word_wrap=True))
            with Horizontal(classes="actions"):
                yield Button("open", classes="open", variant="default", compact=True)
                yield Button("revert", classes="revert", variant="warning", compact=True)

    @on(Button.Pressed, ".revert")
    def _revert(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Revert(self.path))

    @on(Button.Pressed, ".open")
    def _open(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.open_in_editor(self.path)  # type: ignore[attr-defined]

    def mark_reverted(self) -> None:
        self.add_class("reverted")
        self.query_one(Collapsible).title = f"{self.path}  reverted"
        self.query_one(".revert", Button).disabled = True


class PlanCard(TurnWidget):
    DEFAULT_CSS = """
    PlanCard { height: auto; margin: 1 0 0 1; border: round $accent; padding: 0 1; }
    PlanCard .head { height: auto; }
    PlanCard .head Label { width: 1fr; color: $accent; text-style: bold; }
    PlanCard .head Button { min-width: 9; }
    PlanCard Markdown { margin: 0; padding: 0; background: transparent; }
    """

    def __init__(self, path: str, markdown: str, turn: int) -> None:
        super().__init__(turn=turn)
        self.path = path
        self.markdown = markdown

    def compose(self) -> ComposeResult:
        with Horizontal(classes="head"):
            yield Label(f"plan · {self.path}")
            yield Button("build", id="plan-build", variant="success", compact=True)
        yield Markdown(self.markdown)


class SystemNote(TurnWidget):
    DEFAULT_CSS = """
    SystemNote { height: auto; color: $text-muted; padding: 0 1; margin: 0; }
    SystemNote.warning { color: $warning; }
    SystemNote.error { color: $error; }
    """

    def __init__(self, text: str, level: str = "info", turn: int = 0) -> None:
        super().__init__(turn=turn, classes=level)
        self.text = text

    def render(self) -> Text:
        return Text("  " + self.text)


# --- input ---------------------------------------------------------------------

class ChatInput(TextArea):
    """Enter sends, Shift+Enter / Ctrl+J insert a newline."""
    DEFAULT_CSS = """
    ChatInput { height: auto; max-height: 8; min-height: 3; border: tall $primary 40%; background: $surface; }
    ChatInput:focus { border: tall $primary; }
    """

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")


# --- modals --------------------------------------------------------------------

class PermissionModal(ModalScreen[Decision]):
    DEFAULT_CSS = """
    PermissionModal { align: center middle; }
    PermissionModal > Vertical { width: 80; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    PermissionModal .cmd { background: $boost; padding: 0 1; margin: 1 0; }
    PermissionModal Horizontal { height: auto; align-horizontal: right; }
    PermissionModal Button { margin-left: 1; }
    """
    BINDINGS = [Binding("escape", "deny", "Deny"), Binding("y", "allow", "Allow"), Binding("n", "deny", "Deny"), Binding("a", "always", "Always")]

    def __init__(self, req: PermissionRequest, *, offer_always: bool, prefix: str) -> None:
        super().__init__()
        self.req = req
        self.offer_always = offer_always
        self.prefix = prefix

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"⚡ {self.req.tool} wants to run:")
            yield Static(self.req.detail, classes="cmd")
            with Horizontal():
                yield Button("Deny (n)", id="deny", variant="error")
                if self.offer_always:
                    yield Button(f"Always allow '{self.prefix}' (a)", id="always", variant="primary")
                yield Button("Allow once (y)", id="allow", variant="success")

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss({"deny": "deny", "allow": "allow", "always": "allow_always"}[event.button.id])  # type: ignore[index]

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_deny(self) -> None:
        self.dismiss("deny")

    def action_always(self) -> None:
        if self.offer_always:
            self.dismiss("allow_always")


class UndoModal(ModalScreen[str | None]):
    """Returns 'keep', 'revert', or None (cancel)."""
    DEFAULT_CSS = """
    UndoModal { align: center middle; }
    UndoModal > Vertical { width: 80; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    UndoModal .files { color: $text-muted; margin: 1 0; }
    UndoModal Horizontal { height: auto; align-horizontal: right; }
    UndoModal Button { margin-left: 1; }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("k", "keep", "Keep"), Binding("u", "revert", "Undo files")]

    def __init__(self, preview: UndoPreview) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        p = self.preview
        n = len(p.turns)
        with Vertical():
            yield Label(f"↶ Undo {n} turn{'s' if n != 1 else ''}, back to before turn {p.turns[0].number}?")
            yield Static("\n".join(f"  {t.number}. {t.text[:80]}" for t in p.turns))
            yield Static("This changed: " + ", ".join(p.files) + ("\nShell commands from these turns can't be undone." if p.ran_shell else ""), classes="files")
            with Horizontal():
                yield Button("Cancel (esc)", id="cancel")
                yield Button("Keep file changes (k)", id="keep", variant="primary")
                yield Button("Undo file changes (u)", id="revert", variant="warning")

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "cancel" else event.button.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_keep(self) -> None:
        self.dismiss("keep")

    def action_revert(self) -> None:
        self.dismiss("revert")


class HelpModal(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpModal { align: center middle; }
    HelpModal > VerticalScroll { width: 90; height: 80%; border: thick $primary; background: $surface; padding: 1 2; }
    """
    BINDINGS = [Binding("escape", "dismiss_help", "Close")]

    HELP = """\
# toki harness

**Modes** (sidebar, or `/agent` `/ask` `/plan`)
- **agent** reads, edits, and runs; edits apply to disk and show as diff cards
- **ask** read-only; the edit and shell tools are not in the model's schema
- **plan** researches, writes `.harness/plan.md`; press **build** on the plan card

**Shell permissions** (sidebar): none · ask · allowlist · open. **Web**: toggles `web_search` / `fetch_url`.

**Chat**
- Enter sends, Shift+Enter or Ctrl+J inserts a newline, `@path` attaches a file
- **↶ rewind** on any of your messages undoes everything from that turn on; you're asked whether to keep or undo its file changes
- tool cards expand on click; diff cards have **open** and **revert**
- Escape stops a running turn

**Commands** `/undo [n]` `/redo` `/new` `/build` `/sessions` `/changes` `/permissions <level>` `/web on|off` `/help` `/quit`

**Keys** Ctrl+B sidebar · Ctrl+N new conversation · F1 help · Ctrl+Q quit

Edits are reviewed in your editor: with git, Cursor's Source Control view shows every change with per-hunk discard.
Conversations autosave to `.harness/sessions/`; pick one in the sidebar to resume.
"""

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Markdown(self.HELP)

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


# --- sidebar -------------------------------------------------------------------

class Sidebar(Vertical):
    DEFAULT_CSS = """
    Sidebar { width: 34; border-right: solid $panel; padding: 0 1; }
    Sidebar.hidden { display: none; }
    Sidebar .section { color: $text-muted; text-style: bold; margin: 1 0 0 0; }
    Sidebar RadioSet { border: none; background: transparent; padding: 0; height: auto; }
    Sidebar RadioButton { padding: 0; }
    Sidebar Select { margin: 0; width: 100%; }
    Sidebar Horizontal { height: auto; }
    Sidebar Switch { border: none; height: 1; padding: 0; }
    Sidebar .switch-label { height: 1; padding: 0 1; }
    Sidebar ListView { height: auto; max-height: 10; background: transparent; }
    Sidebar ListItem { padding: 0 1; }
    Sidebar .row-btn { min-width: 8; margin: 0 1 0 0; }
    Sidebar .muted { color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Label("mode", classes="section")
        with RadioSet(id="mode"):
            for m in MODE_ORDER:
                yield RadioButton(m.value, value=(m is Mode.AGENT))
        yield Label("shell", classes="section")
        yield Select([(p.value, p.value) for p in PermissionLevel], id="perm", allow_blank=False, value=PermissionLevel.ASK.value, compact=True, tooltip="shell permission level")
        with Horizontal():
            yield Switch(value=True, id="web")
            yield Label("web access", classes="switch-label")
        yield Label("plan", classes="section")
        yield Static("no plan", id="plan-status", classes="muted")
        with Horizontal():
            yield Button("build", id="side-build", classes="row-btn", compact=True, variant="success", disabled=True)
            yield Button("clear", id="side-plan-clear", classes="row-btn", compact=True, disabled=True)
        yield Label("changed files", classes="section")
        yield ListView(id="changes")
        with Horizontal():
            yield Button("revert all", id="side-revert-all", classes="row-btn", compact=True, variant="warning", disabled=True)
        yield Label("conversations", classes="section")
        yield ListView(id="sessions")
        with Horizontal():
            yield Button("new", id="side-new", classes="row-btn", compact=True, variant="primary")


class ChangeItem(ListItem):
    def __init__(self, path: str, kind: str) -> None:
        super().__init__(Label(f"{kind[0].upper()}  {path}"))
        self.path = path


class SessionItem(ListItem):
    def __init__(self, info: SessionInfo, current: bool) -> None:
        mark = "● " if current else "  "
        super().__init__(Label(f"{mark}{info.title[:26]}\n   {info.age} · {info.turns} turns"))
        self.info = info


# --- the app -------------------------------------------------------------------

class HarnessTUI(App[None]):
    TITLE = "toki harness"
    CSS = """
    #main { height: 1fr; }
    #chat { height: 1fr; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
    #chat .welcome { color: $text-muted; padding: 1; }
    """
    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+n", "new_session", "New"),
        Binding("escape", "stop_turn", "Stop", show=False),
        Binding("f1", "help", "Help"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, harness: Harness, *, resume: str | None = None) -> None:
        super().__init__()
        self.harness = harness
        self._resume = resume
        self._busy = False
        self._current_md: AssistantMessage | None = None
        self._tools: dict[str, ToolCard] = {}
        self._turn = 0
        self.prefill = ""

    # ----- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield Sidebar(id="sidebar")
            with Vertical():
                yield VerticalScroll(id="chat")
                yield Static("", id="status")
                yield ChatInput(id="input", placeholder="Message the agent… (Enter sends, Shift+Enter newline, @path attaches, /help)")
        yield Footer()

    def on_mount(self) -> None:
        h = self.harness
        h.ctx.approver = self.approve
        self.sub_title = f"{h.workspace.root}  ·  {describe(h.model)}"
        self._sync_sidebar()
        self.query_one("#input", ChatInput).focus()
        sessions = h.list_sessions()
        if self._resume and self._resume != "new" and sessions:
            idx = 0 if self._resume == "last" else int(self._resume) - 1
            if 0 <= idx < len(sessions):
                self.resume_session(sessions[idx])
                return
        note = "Ask about the workspace, or describe a change. F1 for help."
        if sessions and self._resume != "new":
            note += f"  {len(sessions)} saved conversation{'s' if len(sessions) != 1 else ''} in the sidebar."
        self._chat().mount(Static(note, classes="welcome"))
        self._status()

    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    async def _add(self, widget: Widget) -> Widget:
        chat = self._chat()
        await chat.mount(widget)
        chat.scroll_end(animate=False)
        return widget

    def _status(self) -> None:
        h = self.harness
        n = len(h.changes())
        parts = [
            f"[b]{h.mode.value}[/b]",
            f"shell: {h.permissions.level.value}",
            "web: " + ("on" if h.permissions.web else "off"),
            f"{n} file{'s' if n != 1 else ''} changed" if n else "",
            f"{h.usage.total_tokens:,} tokens" if h.usage.total_tokens else "",
            "working… (Esc stops)" if self._busy else "",
        ]
        self.query_one("#status", Static).update("  │  ".join(p for p in parts if p))

    def _sync_sidebar(self) -> None:
        """Schedule a sidebar refresh (safe from sync handlers)."""
        self.call_later(self._sync_sidebar_async)

    async def _sync_sidebar_async(self) -> None:
        h = self.harness
        radios = self.query_one("#mode", RadioSet)
        buttons = list(radios.query(RadioButton))
        for btn, m in zip(buttons, MODE_ORDER):
            if m is h.mode and not btn.value:
                with radios.prevent(RadioSet.Changed):
                    btn.value = True
        perm = self.query_one("#perm", Select)
        with perm.prevent(Select.Changed):
            perm.value = h.permissions.level.value
        web = self.query_one("#web", Switch)
        with web.prevent(Switch.Changed):
            web.value = h.permissions.web
        # plan
        plan = h.plan_text()
        title = next((l.lstrip("# ").strip() for l in (plan or "").splitlines() if l.startswith("#")), None)
        self.query_one("#plan-status", Static).update(title or (".harness/plan.md" if plan else "no plan"))
        self.query_one("#side-build", Button).disabled = plan is None
        self.query_one("#side-plan-clear", Button).disabled = plan is None
        # changes
        changes = self.query_one("#changes", ListView)
        await changes.clear()
        await changes.extend([ChangeItem(rel, kind) for rel, kind in h.changes()])
        self.query_one("#side-revert-all", Button).disabled = not h.changes()
        # sessions
        sessions = self.query_one("#sessions", ListView)
        await sessions.clear()
        await sessions.extend([SessionItem(info, info.path == h.session_path) for info in h.list_sessions()[:8]])
        self._status()

    # ----- turns ---------------------------------------------------------------

    @on(ChatInput.Submitted)
    async def _submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text
        if text.startswith("/"):
            await self.command(text)
            return
        await self.send(text)

    async def send(self, text: str, *, echo: str | None = None) -> None:
        if self._busy:
            self.notify("still working on the previous message (Esc to stop)", severity="warning")
            return
        self._turn = len(self.harness.history()) + 1
        attachments = self._mentions(text)
        await self._add(UserMessage(echo or text, self._turn))
        if attachments:
            await self._add(SystemNote("attached: " + ", ".join(attachments), turn=self._turn))
        self._busy = True
        self._status()
        self._run_turn(text, attachments)

    def _mentions(self, text: str) -> list[str]:
        return find_mentions(self.harness.workspace, text)

    @work(thread=True, exclusive=True, group="turn")
    def _run_turn(self, text: str, attachments: list[str]) -> None:
        try:
            for ev in self.harness.run_turn(text, attachments):
                self.call_from_thread(self._handle_event, ev)
        except Exception as e:  # provider errors: keep the app alive
            self.harness.abort()
            self.call_from_thread(self._handle_event, Notice(f"{type(e).__name__}: {e}", "error"))
            self.call_from_thread(self._handle_event, TurnEnded("", None, cancelled=True))

    async def _handle_event(self, ev) -> None:
        turn = self._turn
        if isinstance(ev, TextDelta):
            if self._current_md is None:
                self._current_md = await self._add(AssistantMessage(turn))  # type: ignore[assignment]
            await self._current_md.append(ev.text)
            self._chat().scroll_end(animate=False)
        elif isinstance(ev, ThinkingDelta):
            if self._current_md is None:
                self._current_md = await self._add(AssistantMessage(turn))  # type: ignore[assignment]
            await self._current_md.append_thinking(ev.text)
        elif isinstance(ev, ToolStarted):
            await self._close_md()
            card = ToolCard(ev.call_id, ev.name, turn, args=ev.args)
            self._tools[ev.call_id] = card
            await self._add(card)
        elif isinstance(ev, ToolArgDelta):
            card = self._tools.get(ev.call_id)
            if card:
                card.arg_delta(ev.arg, ev.text)
        elif isinstance(ev, ToolFinished):
            card = self._tools.get(ev.call_id)
            if card:
                card.finish(ev.summary, ev.output, ev.ok)
        elif isinstance(ev, FileChanged):
            await self._add(DiffCard(ev.path, ev.kind, ev.diff, turn))
            self._sync_sidebar()
        elif isinstance(ev, PlanUpdated):
            await self._add(PlanCard(ev.path, ev.markdown, turn))
            self._sync_sidebar()
        elif isinstance(ev, Notice):
            await self._add(SystemNote(ev.text, ev.level, turn=turn))
        elif isinstance(ev, TurnEnded):
            await self._close_md()
            if ev.cancelled:
                await self._add(SystemNote("⏹ interrupted", "warning", turn=turn))
            self._busy = False
            self._tools.clear()
            self._sync_sidebar()

    async def _close_md(self) -> None:
        if self._current_md is not None:
            await self._current_md.finish()
            self._current_md = None

    # ----- approver (called on the worker thread) -------------------------------

    def approve(self, req: PermissionRequest) -> Decision:
        from ..core.tools import _prefix_of
        done = threading.Event()
        result: dict[str, Decision] = {}

        def callback(decision: Decision | None) -> None:
            result["d"] = decision or "deny"
            done.set()

        offer_always = self.harness.permissions.level is PermissionLevel.ALLOWLIST
        self.call_from_thread(self.push_screen, PermissionModal(req, offer_always=offer_always, prefix=_prefix_of(req.detail)), callback)
        done.wait()
        return result.get("d", "deny")

    # ----- undo / redo -----------------------------------------------------------

    @on(UserMessage.Rewind)
    async def _rewind_clicked(self, event: UserMessage.Rewind) -> None:
        n = len(self.harness.history()) - event.turn + 1
        await self.undo(n)

    async def undo(self, n: int = 1) -> None:
        if self._busy:
            self.notify("stop the running turn first (Esc)", severity="warning")
            return
        preview = self.harness.undo_preview(n)
        if preview is None:
            self.notify("nothing to undo")
            return

        def apply(choice: str | None) -> None:
            if choice is None:
                return
            self._apply_undo(n, revert=(choice == "revert"))

        if preview.files and not preview.resumed:
            self.push_screen(UndoModal(preview), apply)
        else:
            if preview.resumed and preview.files:
                self.notify("loaded from a saved session: file changes stay, messages are removed")
            self._apply_undo(n, revert=False)

    def _apply_undo(self, n: int, *, revert: bool) -> None:
        result = self.harness.undo(n, revert_files=revert)
        if result is None:
            return
        first = result.turns[0].number
        for w in list(self._chat().query(TurnWidget)):
            if w.turn >= first:
                w.remove()
        msg = f"undid {len(result.turns)} turn{'s' if len(result.turns) != 1 else ''}"
        if result.restored:
            msg += "; restored " + ", ".join(result.restored)
        if result.kept:
            msg += "; kept changes to " + ", ".join(result.kept)
        if result.ran_shell:
            msg += " (shell commands can't be undone)"
        self.notify(msg + "  ·  /redo brings it back", timeout=6)
        inp = self.query_one("#input", ChatInput)
        inp.load_text(result.turns[0].text)
        inp.focus()
        self._sync_sidebar()

    async def redo(self) -> None:
        result = self.harness.redo()
        if result is None:
            self.notify("nothing to redo")
            return
        start = result.turns[0].start
        await self._render_messages(self.harness.agent.messages[start:], result.turns)
        self.notify(f"restored {len(result.turns)} turn{'s' if len(result.turns) != 1 else ''}" + ("; re-applied " + ", ".join(result.restored) if result.restored else ""))
        self.query_one("#input", ChatInput).clear()
        self._sync_sidebar()

    async def _render_messages(self, messages: list[TokiMessage], turns: list[TurnRecord]) -> None:
        """Rebuild chat widgets from saved messages (resume, redo)."""
        turn_no = 0
        by_start = {t.start: t for t in turns}
        offset = self.harness.agent.messages.index(messages[0]) if messages else 0
        outputs = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
        for i, m in enumerate(messages):
            rec = by_start.get(offset + i)
            if m.role == "user":
                turn_no = rec.number if rec else turn_no + 1
                await self._add(UserMessage(m.content.split("\n\n<attached", 1)[0], turn_no))
            elif m.role == "assistant":
                if m.content.strip():
                    await self._add(AssistantMessage(turn_no, text=m.content))
                for tc in m.tool_calls or []:
                    card = ToolCard(tc.id, tc.function.name, turn_no, args=tc.function.arguments)
                    await self._add(card)
                    out = outputs.get(tc.id, "")
                    card.finish(out.splitlines()[0][:80] if out else "", out, not out.startswith("error"))
        self._chat().scroll_end(animate=False)

    # ----- sessions ---------------------------------------------------------------

    def resume_session(self, info: SessionInfo) -> None:
        self.run_worker(self._resume_async(info), exclusive=True, group="turn")

    async def _resume_async(self, info: SessionInfo) -> None:
        try:
            messages = self.harness.resume_session(info.path)
        except Exception as e:
            self.notify(f"could not load {info.path.name}: {e}", severity="error")
            return
        await self._chat().remove_children()
        await self._add(SystemNote(f"▸ resumed “{info.title}”  ({info.turns} turns, {info.age})"))
        await self._render_messages(messages, self.harness.history())
        self._sync_sidebar()

    # ----- sidebar handlers -----------------------------------------------------

    @on(RadioSet.Changed, "#mode")
    def _mode_changed(self, event: RadioSet.Changed) -> None:
        self.harness.set_mode(MODE_ORDER[event.radio_set.pressed_index])
        self._status()

    @on(Select.Changed, "#perm")
    def _perm_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.harness.set_permission_level(PermissionLevel(str(event.value)))
            self._status()

    @on(Switch.Changed, "#web")
    def _web_changed(self, event: Switch.Changed) -> None:
        self.harness.set_web(bool(event.value))
        self._status()

    @on(Button.Pressed, "#side-build")
    @on(Button.Pressed, "#plan-build")
    async def _build(self, event: Button.Pressed) -> None:
        event.stop()
        prompt = self.harness.build_prompt()
        if prompt is None:
            self.notify("no plan to build", severity="warning")
            return
        self._sync_sidebar()
        await self.send(prompt, echo="Implement the plan")

    @on(Button.Pressed, "#side-plan-clear")
    def _plan_clear(self, event: Button.Pressed) -> None:
        archived = self.harness.clear_plan()
        self.notify(f"plan archived to {archived}" if archived else "no plan")
        self._sync_sidebar()

    @on(Button.Pressed, "#side-revert-all")
    def _revert_all(self, event: Button.Pressed) -> None:
        done = self.harness.revert_all()
        for card in self._chat().query(DiffCard):
            card.mark_reverted()
        self.notify("; ".join(done) if done else "nothing to revert")
        self._sync_sidebar()

    @on(DiffCard.Revert)
    def _revert_one(self, event: DiffCard.Revert) -> None:
        try:
            self.notify(self.harness.revert(event.path))
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        for card in self._chat().query(DiffCard):
            if card.path == event.path:
                card.mark_reverted()
        self._sync_sidebar()

    @on(ListView.Selected, "#changes")
    def _change_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ChangeItem):
            self.open_in_editor(item.path)

    @on(ListView.Selected, "#sessions")
    def _session_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, SessionItem) and item.info.path != self.harness.session_path:
            if self._busy:
                self.notify("stop the running turn first (Esc)", severity="warning")
                return
            self.resume_session(item.info)

    @on(Button.Pressed, "#side-new")
    async def _new_clicked(self, event: Button.Pressed) -> None:
        await self.action_new_session()

    def open_in_editor(self, path: str) -> None:
        import shutil
        import subprocess
        exe = next((e for e in ("cursor", "code") if shutil.which(e)), None)
        if exe is None:
            self.notify("neither `cursor` nor `code` is on PATH", severity="warning")
            return
        subprocess.Popen([exe, "-g", str(self.harness.workspace.resolve(path))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ----- slash commands -------------------------------------------------------

    async def command(self, line: str) -> None:
        name, _, args = line[1:].partition(" ")
        name = name.lower()
        h = self.harness
        if name in ("help", "?"):
            self.push_screen(HelpModal())
        elif name in ("agent", "ask", "plan"):
            h.set_mode(Mode(name))
            self._sync_sidebar()
            if name == "plan" and h.plan_text():
                await self._add(PlanCard(".harness/plan.md", h.plan_text() or "", 0))
        elif name == "mode" and args.strip() in ("agent", "ask", "plan"):
            h.set_mode(Mode(args.strip()))
            self._sync_sidebar()
        elif name == "build":
            prompt = h.build_prompt()
            if prompt is None:
                self.notify("no plan to build", severity="warning")
            else:
                self._sync_sidebar()
                await self.send(prompt, echo="Implement the plan")
        elif name == "undo":
            await self.undo(int(args) if args.strip().isdigit() else 1)
        elif name == "redo":
            await self.redo()
        elif name == "new":
            await self.action_new_session()
        elif name in ("permissions", "perms") and args.strip():
            try:
                h.set_permission_level(PermissionLevel(args.strip()))
                self._sync_sidebar()
            except ValueError:
                self.notify("levels: none, ask, allowlist, open", severity="warning")
        elif name == "allow" and args.strip():
            h.allow(args.strip())
            self.notify(f"allowed {args.strip()!r}")
        elif name == "web" and args.strip() in ("on", "off"):
            h.set_web(args.strip() == "on")
            self._sync_sidebar()
        elif name in ("sessions", "changes"):
            self.query_one("#sidebar").remove_class("hidden")
            self.notify("see the sidebar")
        elif name in ("quit", "exit", "q"):
            self.exit()
        else:
            self.notify(f"unknown command /{name}; F1 for help", severity="warning")

    # ----- actions -------------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        self.query_one("#sidebar").toggle_class("hidden")

    async def action_new_session(self) -> None:
        if self._busy:
            self.notify("stop the running turn first (Esc)", severity="warning")
            return
        self.harness.reset()
        await self._chat().remove_children()
        await self._add(SystemNote("new conversation; the previous one is saved in the sidebar"))
        self._sync_sidebar()

    def action_stop_turn(self) -> None:
        if self._busy:
            self.harness.cancel()
            self.notify("stopping…")

    def action_help(self) -> None:
        self.push_screen(HelpModal())


def run(root: str | Path, model_spec: str, *, mode: Mode, permission_level: PermissionLevel | None, thinking: bool,
        resume: str | None = None) -> int:
    from ..core import make_model
    try:
        model = make_model(model_spec)
    except Exception as e:
        print(f"could not load model {model_spec!r}: {e}\nset the provider's API key, pass --model provider:name, or try --model demo")
        return 2
    harness = Harness(model, root, mode=mode, permission_level=permission_level, capture_thinking=thinking)
    HarnessTUI(harness, resume=resume).run()
    return 0
