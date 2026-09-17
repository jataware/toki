"""
Slash commands. One table drives dispatch, `/help`, and tab completion, so
they never drift apart. Handlers receive the running `App` and the argument
string; they return nothing and print through the app's console.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from rich.table import Table
from rich.text import Text

from ..core import Mode, PermissionLevel, WorkspaceError, describe, make_model

if TYPE_CHECKING:
    from .app import App


@dataclass
class Command:
    name: str
    handler: Callable[["App", str], None]
    short: str
    usage: str = ""
    long: str = ""
    section: str = "General"
    aliases: list[str] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)   # for tab completion of the first argument


def _help(app: "App", args: str) -> None:
    from .ui import help_table
    help_table(app.console, COMMANDS, args.strip() or None)


def _mode(app: "App", args: str) -> None:
    arg = args.strip().lower()
    if not arg:
        app.show_status()
        return
    try:
        app.set_mode(Mode(arg))
    except ValueError:
        app.error(f"unknown mode {arg!r}; choose agent, ask, or plan")


def _agent(app: "App", args: str) -> None:
    app.set_mode(Mode.AGENT)


def _ask(app: "App", args: str) -> None:
    app.set_mode(Mode.ASK)


def _plan(app: "App", args: str) -> None:
    sub, _, rest = args.strip().partition(" ")
    sub = sub.lower()
    h = app.harness
    if sub in ("", "show"):
        if sub == "":
            app.set_mode(Mode.PLAN)
        text = h.plan_text()
        if text:
            app.renderer.plan_panel(text, h.workspace.relpath(h.plan_path))
        elif sub == "show":
            app.info("no plan yet. In plan mode, describe what you want and the agent will write one.")
        else:
            app.info("describe what you want; the agent will research and write .harness/plan.md for you to review.")
    elif sub == "edit":
        if not h.plan_path.exists():
            app.error("no plan to edit yet")
            return
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("cursor -w" if shutil.which("cursor") else "vi")
        app.info(f"opening {h.workspace.relpath(h.plan_path)} with {editor} …")
        try:
            subprocess.run([*shlex.split(editor), str(h.plan_path)], check=False)
        except OSError as e:
            app.error(f"could not launch editor: {e}")
            return
        app.renderer.plan_panel(h.plan_text() or "", h.workspace.relpath(h.plan_path))
    elif sub == "clear":
        archived = h.clear_plan()
        app.info(f"plan archived to {archived}" if archived else "no plan to clear")
    else:
        app.error("usage: /plan [show|edit|clear]")


def _build(app: "App", args: str) -> None:
    if app.harness.plan_text() is None:
        app.error("no plan to build. Switch to /plan and describe what you want first.")
        return
    app.set_mode(Mode.AGENT)
    prompt = app.harness.build_prompt()
    assert prompt is not None
    extra = args.strip()
    app.run_turn(prompt + (f"\n\nAdditional instructions from the user: {extra}" if extra else ""), echo="Implement the plan" + (f" ({extra})" if extra else ""))


def _changes(app: "App", args: str) -> None:
    changes = app.harness.changes()
    if not changes:
        app.info("no files changed this session")
        return
    table = Table(box=None, padding=(0, 2), show_header=False, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    for rel, kind in changes:
        diff = app.harness.diff(rel)
        plus = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        minus = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        table.add_row(kind, Text.assemble((rel, "bold"), (f"  +{plus} ", "green"), (f"-{minus}", "red")))
    app.console.print(table)
    app.info("review in your editor (git view), or /diff <path>, /revert <path|all>, /open <path>")


def _diff(app: "App", args: str) -> None:
    rel = args.strip()
    targets = [rel] if rel else [r for r, _ in app.harness.changes()]
    if not targets:
        app.info("no files changed this session")
        return
    for r in targets:
        try:
            diff = app.harness.diff(r)
        except WorkspaceError as e:
            app.error(str(e))
            continue
        if not diff:
            app.info(f"{r}: no changes")
            continue
        app.renderer.diff_panel(r, diff, max_lines=10_000)


def _revert(app: "App", args: str) -> None:
    rel = args.strip()
    if not rel:
        app.error("usage: /revert <path> | /revert all")
        return
    try:
        if rel == "all":
            done = app.harness.revert_all()
            app.info("; ".join(done) if done else "nothing to revert")
        else:
            app.info(app.harness.revert(rel))
    except WorkspaceError as e:
        app.error(str(e))


def _undo(app: "App", args: str) -> None:
    arg = args.strip()
    n = int(arg) if arg.isdigit() else 1
    preview = app.harness.undo_preview(n)
    if preview is None:
        app.info("nothing to undo")
        return
    for t in preview.turns:
        app.console.print(Text.assemble(("  ↶ turn ", "yellow"), (str(t.number), "bold yellow"), ("  " + t.text[:120], "italic")))
    revert = False
    if preview.files:
        if preview.resumed:
            app.info("this turn was loaded from a saved session, so its file changes can't be reverted; the messages will be removed")
        else:
            app.console.print(Text("  this turn changed: " + ", ".join(preview.files), style="dim"))
            answer = app.ask_choice("  [k] keep the file changes   [u] undo them   (Ctrl-C cancels) › ", {"k": "keep", "u": "undo"})
            if answer is None:
                app.info("cancelled; nothing changed")
                return
            revert = answer == "undo"
    if preview.ran_shell:
        app.info("note: shell commands from this turn can't be undone")
    result = app.harness.undo(n, revert_files=revert)
    assert result is not None
    parts = [f"removed {len(result.turns)} turn{'s' if len(result.turns) != 1 else ''}"]
    if result.restored:
        parts.append("restored " + ", ".join(result.restored))
    if result.kept:
        parts.append("kept changes to " + ", ".join(result.kept))
    app.info("; ".join(parts) + ". /redo brings it back")
    app.prefill = result.turns[0].text
    app.completer.refresh()


def _redo(app: "App", args: str) -> None:
    result = app.harness.redo()
    if result is None:
        app.info("nothing to redo")
        return
    for t in result.turns:
        app.console.print(Text.assemble(("  ↷ turn ", "green"), (str(t.number), "bold green"), ("  " + t.text[:120], "italic")))
    msg = f"restored {len(result.turns)} turn{'s' if len(result.turns) != 1 else ''}"
    if result.restored:
        msg += "; re-applied changes to " + ", ".join(result.restored)
    app.info(msg)
    app.prefill = ""
    app.completer.refresh()


def _history(app: "App", args: str) -> None:
    turns = app.harness.history()
    if not turns:
        app.info("no turns yet")
        return
    table = Table(box=None, padding=(0, 2), show_header=False, pad_edge=False)
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_column(style="dim")
    for t in turns:
        extra = []
        if t.files:
            extra.append(f"{len(t.files)} file{'s' if len(t.files) != 1 else ''}")
        if t.ran_shell:
            extra.append("shell")
        if t.resumed:
            extra.append("resumed")
        table.add_row(str(t.number), Text(t.text[:100], style="italic"), " · ".join(extra))
    app.console.print(table)
    app.info("/undo removes the last turn, /undo 3 the last three" + ("; /redo is available" if app.harness.can_redo else ""))


def _open(app: "App", args: str) -> None:
    target = args.strip()
    if not target:
        app.error("usage: /open <path[:line]>")
        return
    exe = next((e for e in ("cursor", "code") if shutil.which(e)), None)
    if exe is None:
        app.error("neither `cursor` nor `code` is on PATH")
        return
    path, _, line = target.partition(":")
    try:
        abs_path = app.harness.workspace.resolve(path)
    except WorkspaceError as e:
        app.error(str(e))
        return
    loc = f"{abs_path}:{line}" if line else str(abs_path)
    subprocess.Popen([exe, "-g", loc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    app.info(f"opened {target} in {exe}")


def _permissions(app: "App", args: str) -> None:
    arg = args.strip().lower()
    perms = app.harness.permissions
    if not arg:
        app.console.print(Text.assemble(("  shell: ", "dim"), (perms.level.value, "bold"), ("  " + perms.level.blurb, "dim")))
        if perms.allowlist:
            app.console.print(Text("  allowlist: " + ", ".join(perms.allowlist), style="dim"))
        return
    try:
        app.harness.set_permission_level(PermissionLevel(arg))
    except ValueError:
        app.error(f"unknown level {arg!r}; choose none, ask, allowlist, or open")
        return
    app.info(f"shell permissions: {arg} · {app.harness.permissions.level.blurb}")
    if app.harness.permissions.level is PermissionLevel.ALLOWLIST and not perms.allowlist:
        app.info("allowlist is empty; add prefixes with /allow <command prefix>")


def _allow(app: "App", args: str) -> None:
    prefix = args.strip()
    if not prefix:
        app.info("allowlist: " + (", ".join(app.harness.permissions.allowlist) or "(empty)"))
        return
    added = app.harness.allow(prefix)
    app.info(f"allowed {prefix!r}" if added else f"{prefix!r} already allowed")
    if app.harness.permissions.level is not PermissionLevel.ALLOWLIST:
        app.info("note: the allowlist only applies with /permissions allowlist")


def _disallow(app: "App", args: str) -> None:
    prefix = args.strip()
    if not prefix:
        app.error("usage: /disallow <prefix>")
        return
    app.info(f"removed {prefix!r}" if app.harness.disallow(prefix) else f"{prefix!r} was not in the allowlist")


def _web(app: "App", args: str) -> None:
    arg = args.strip().lower()
    if arg in ("on", "off"):
        app.harness.set_web(arg == "on")
    state = "on" if app.harness.permissions.web else "off"
    app.info(f"web access: {state}" + ("  (web_search and fetch_url available)" if state == "on" else "  (no internet tools)"))


def _sessions(app: "App", args: str) -> None:
    sessions = app.harness.list_sessions()
    if not sessions:
        app.info("no saved conversations in this workspace yet")
        return
    app.session_table(sessions)
    app.info("resume with /resume <n>; the current conversation is saved automatically")


def _resume(app: "App", args: str) -> None:
    arg = args.strip()
    sessions = app.harness.list_sessions()
    if not sessions:
        app.info("no saved conversations in this workspace yet")
        return
    if not arg:
        app.session_table(sessions)
        app.info("usage: /resume <n>")
        return
    try:
        idx = int(arg) - 1
        chosen = sessions[idx]
        if idx < 0:
            raise IndexError
    except (ValueError, IndexError):
        app.error(f"pick a number between 1 and {len(sessions)}")
        return
    app.resume(chosen)


def _model(app: "App", args: str) -> None:
    spec = args.strip()
    if not spec:
        app.info(f"model: {describe(app.harness.model)}")
        return
    try:
        model = make_model(spec)
    except Exception as e:  # missing key, bad provider, ...
        app.error(f"could not load {spec!r}: {e}")
        return
    app.harness.agent.model = model
    app.info(f"model: {describe(model)} (conversation kept)")


def _new(app: "App", args: str) -> None:
    app.harness.reset()
    app.info("new conversation started; the previous one is saved (/sessions)")


def _tokens(app: "App", args: str) -> None:
    u = app.harness.usage
    t = Text.assemble(
        ("  turns ", "dim"), (str(app.harness.turns), "bold"),
        ("   prompt ", "dim"), (f"{u.prompt_tokens:,}", "bold"),
        ("   completion ", "dim"), (f"{u.completion_tokens:,}", "bold"),
        ("   total ", "dim"), (f"{u.total_tokens:,}", "bold"),
    )
    if u.cache_read_tokens or u.cache_write_tokens:
        t.append(f"   cache read {u.cache_read_tokens:,} / write {u.cache_write_tokens:,}", style="dim")
    app.console.print(t)
    app.info(f"{len(app.harness.agent.messages)} messages in context; /new clears the conversation")


def _tree(app: "App", args: str) -> None:
    try:
        app.console.print(Text(app.harness.workspace.tree(args.strip() or ".", depth=2), style="dim"))
    except WorkspaceError as e:
        app.error(str(e))


def _think(app: "App", args: str) -> None:
    arg = args.strip().lower()
    if arg in ("on", "off"):
        app.harness.capture_thinking = arg == "on"
    app.info(f"thinking display: {'on' if app.harness.capture_thinking else 'off'}")


def _quit(app: "App", args: str) -> None:
    app.running = False


COMMANDS: list[Command] = [
    Command("help", _help, "Show this help, or details for one command", "[command]"),
    Command("mode", _mode, "Show or set the mode", "[agent|ask|plan]", section="Modes", choices=["agent", "ask", "plan"]),
    Command("agent", _agent, "Agent mode: read, edit, and run", section="Modes"),
    Command("ask", _ask, "Ask mode: read-only questions and explanations", section="Modes"),
    Command("plan", _plan, "Plan mode, or manage the plan file", "[show|edit|clear]", section="Modes", choices=["show", "edit", "clear"],
            long="With no argument, switches to plan mode and shows the current plan. `show` prints it, `edit` opens it in $EDITOR, `clear` archives it."),
    Command("build", _build, "Implement the current plan in agent mode", "[extra instructions]", section="Modes"),
    Command("undo", _undo, "Remove the last turn (or n turns); asks whether to keep its file changes", "[n]", section="Changes"),
    Command("redo", _redo, "Bring back the last undone turn", section="Changes"),
    Command("history", _history, "List the turns in this conversation", section="Changes"),
    Command("changes", _changes, "List files changed this session", section="Changes"),
    Command("diff", _diff, "Show the diff for one changed file, or all", "[path]", section="Changes"),
    Command("revert", _revert, "Undo the session's changes to a file, or all", "<path|all>", section="Changes"),
    Command("open", _open, "Open a file in Cursor (or VS Code)", "<path[:line]>", section="Changes"),
    Command("permissions", _permissions, "Show or set shell permissions", "[none|ask|allowlist|open]", section="Permissions",
            choices=["none", "ask", "allowlist", "open"], aliases=["perms"]),
    Command("allow", _allow, "Add a command prefix to the allowlist", "<prefix>", section="Permissions"),
    Command("disallow", _disallow, "Remove a prefix from the allowlist", "<prefix>", section="Permissions"),
    Command("web", _web, "Show or toggle internet access (web_search, fetch_url)", "[on|off]", section="Permissions", choices=["on", "off"]),
    Command("sessions", _sessions, "List saved conversations in this workspace", section="Session"),
    Command("resume", _resume, "Resume a saved conversation", "<n>", section="Session"),
    Command("model", _model, "Show or switch the model", "[provider:name]", section="Session"),
    Command("new", _new, "Start a fresh conversation (the old one stays saved)", section="Session", aliases=["clear"]),
    Command("tokens", _tokens, "Show token usage for this session", section="Session", aliases=["usage"]),
    Command("tree", _tree, "Print the workspace tree", "[path]", section="Session"),
    Command("think", _think, "Toggle display of model reasoning", "[on|off]", section="Session", choices=["on", "off"]),
    Command("quit", _quit, "Exit", section="Session", aliases=["exit", "q"]),
]

COMMAND_INDEX: dict[str, Command] = {}
for _c in COMMANDS:
    COMMAND_INDEX[_c.name] = _c
    for _a in _c.aliases:
        COMMAND_INDEX[_a] = _c


def dispatch(app: "App", line: str) -> bool:
    """Run `line` as a slash command. Returns False if it isn't one."""
    if not line.startswith("/"):
        return False
    name, _, args = line[1:].partition(" ")
    cmd = COMMAND_INDEX.get(name.lower())
    if cmd is None:
        matches = {c.name: c for c in COMMANDS if c.name.startswith(name.lower())}
        if len(matches) == 1:
            cmd = next(iter(matches.values()))
        elif matches:
            app.error(f"/{name} is ambiguous: " + ", ".join("/" + n for n in matches))
            return True
        else:
            app.error(f"unknown command /{name}. Try /help.")
            return True
    cmd.handler(app, args)
    return True

