"""
Tests for the harness engine. Run from the repo root:

    pytest examples/harness/tests -q

They use `ScriptedModel`, so no API key or network is needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from examples.harness.core import (
    FileChanged,
    Harness,
    Mode,
    PermissionLevel,
    PermissionRequest,
    Permissions,
    PlanUpdated,
    TextDelta,
    ToolArgDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    Workspace,
    WorkspaceError,
)
from examples.harness.testing import ScriptedModel, Turn


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("# Scratch\n\nhello\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    return tmp_path


def events(harness: Harness, text: str) -> list:
    return list(harness.run_turn(text))


def names(evs: list) -> list[str]:
    return [type(e).__name__ for e in evs]


# --- workspace ----------------------------------------------------------------

def test_workspace_jails_paths(ws: Path) -> None:
    w = Workspace(ws)
    assert w.resolve("src/calc.py") == ws / "src" / "calc.py"
    with pytest.raises(WorkspaceError):
        w.resolve("../outside")
    with pytest.raises(WorkspaceError):
        w.resolve("/etc/passwd")


def test_tree_skips_noise(ws: Path) -> None:
    out = Workspace(ws).tree(".", depth=2)
    assert "src/" in out and "calc.py" in out
    assert "node_modules" not in out


def test_snapshot_diff_revert(ws: Path) -> None:
    w = Workspace(ws)
    w.write("src/calc.py", "def add(a, b):\n    return a - b\n")
    assert w.changed_files() == [("src/calc.py", "modified")]
    assert "-    return a + b" in w.diff("src/calc.py")
    w.write("new.txt", "hi\n")
    assert ("new.txt", "created") in w.changed_files()
    assert w.revert("src/calc.py") == "restored src/calc.py"
    assert (ws / "src/calc.py").read_text().endswith("a + b\n")
    assert w.revert("new.txt") == "removed new.txt"
    assert not (ws / "new.txt").exists()
    assert w.changed_files() == []


# --- permissions --------------------------------------------------------------

def test_permission_levels() -> None:
    p = Permissions(PermissionLevel.NONE)
    assert p.check("ls") == "blocked"
    p.level = PermissionLevel.OPEN
    assert p.check("rm -rf /") == "allowed"
    p.level = PermissionLevel.ASK
    p.allow("ls")
    assert p.check("ls") == "ask"  # allowlist ignored at `ask`
    p.level = PermissionLevel.ALLOWLIST
    assert p.check("ls -la src") == "allowed"
    assert p.check("ls && rm x") == "ask"   # every segment must match
    p.allow("git status")
    assert p.check("git status --short") == "allowed"
    assert p.check("git push") == "ask"


def test_permissions_roundtrip(tmp_path: Path) -> None:
    p = Permissions(PermissionLevel.ALLOWLIST, ["pytest"])
    p.save(tmp_path / "c.json")
    q = Permissions.load(tmp_path / "c.json")
    assert q.level is PermissionLevel.ALLOWLIST and q.allowlist == ["pytest"]


# --- harness: modes and tool sets ----------------------------------------------

def test_tool_sets_by_mode_and_level(ws: Path) -> None:
    h = Harness(ScriptedModel([]), ws, permission_level=PermissionLevel.ASK)
    web = {"web_search", "fetch_url"}
    read = {"list_dir", "read_file", "grep", "find_files"}
    agent = {t.name for t in h.tools_for(Mode.AGENT)}
    assert read | web | {"edit_file", "create_file", "delete_file", "write_plan", "run_command"} == agent
    assert {t.name for t in h.tools_for(Mode.ASK)} == read | web
    assert {t.name for t in h.tools_for(Mode.PLAN)} == read | web | {"write_plan"}
    h.set_permission_level(PermissionLevel.NONE)
    assert "run_command" not in {t.name for t in h.tools_for(Mode.AGENT)}
    assert "SHELL: none" in h.system_prompt()


def test_ask_mode_blocks_edits_even_if_model_tries(ws: Path) -> None:
    model = ScriptedModel([
        Turn("editing", [("edit_file", {"path": "README.md", "old": "hello", "new": "bye"})]),
        Turn("done"),
    ])
    h = Harness(model, ws, mode=Mode.ASK)
    evs = events(h, "change it")
    fin = next(e for e in evs if isinstance(e, ToolFinished))
    assert not fin.ok and "not available in ask mode" in fin.output
    assert (ws / "README.md").read_text() == "# Scratch\n\nhello\n"
    assert not any(isinstance(e, FileChanged) for e in evs)


# --- harness: the tool loop -----------------------------------------------------

def test_edit_streams_args_and_emits_diff(ws: Path) -> None:
    model = ScriptedModel([
        Turn("Fixing.", [("edit_file", {"path": "src/calc.py", "old": "a + b", "new": "a * b"})]),
        Turn("Done."),
    ])
    h = Harness(model, ws)
    evs = events(h, "make add multiply")
    assert names(evs)[:1] == ["TextDelta"]
    started = next(e for e in evs if isinstance(e, ToolStarted))
    assert started.name == "edit_file" and started.args == {}          # streaming tool: args arrive later
    deltas = [e for e in evs if isinstance(e, ToolArgDelta)]
    assert "".join(d.text for d in deltas if d.arg == "new") == "a * b"
    fin = next(e for e in evs if isinstance(e, ToolFinished))
    assert fin.ok and fin.args == {"path": "src/calc.py", "old": "a + b", "new": "a * b"}
    changed = next(e for e in evs if isinstance(e, FileChanged))
    assert changed.kind == "modified" and "+    return a * b" in changed.diff
    assert (ws / "src/calc.py").read_text() == "def add(a, b):\n    return a * b\n"
    assert isinstance(evs[-1], TurnEnded) and evs[-1].text == "Done."
    assert h.changes() == [("src/calc.py", "modified")]
    # history is consistent: assistant tool call followed by its tool result
    roles = [m.role for m in h.agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_edit_requires_unique_match(ws: Path) -> None:
    (ws / "dup.txt").write_text("x\nx\n")
    model = ScriptedModel([
        Turn("", [("edit_file", {"path": "dup.txt", "old": "x\n", "new": "y\n"})]),
        Turn("", [("edit_file", {"path": "dup.txt", "old": "zzz", "new": "y"})]),
        Turn("ok"),
    ])
    h = Harness(model, ws)
    fins = [e for e in events(h, "go") if isinstance(e, ToolFinished)]
    assert "occurs 2 times" in fins[0].output
    assert "not found" in fins[1].output
    assert (ws / "dup.txt").read_text() == "x\nx\n"


def test_create_and_delete(ws: Path) -> None:
    model = ScriptedModel([
        Turn("", [("create_file", {"path": "docs/notes.md", "content": "# notes\n"})]),
        Turn("", [("delete_file", {"path": "README.md"})]),
        Turn("ok"),
    ])
    h = Harness(model, ws)
    evs = events(h, "go")
    kinds = [e.kind for e in evs if isinstance(e, FileChanged)]
    assert kinds == ["created", "deleted"]
    assert (ws / "docs/notes.md").exists() and not (ws / "README.md").exists()
    assert sorted(h.changes()) == [("README.md", "deleted"), ("docs/notes.md", "created")]
    h.revert_all()
    assert (ws / "README.md").exists() and not (ws / "docs/notes.md").exists()


def test_run_command_permissions(ws: Path) -> None:
    asked: list[PermissionRequest] = []

    def approver(req: PermissionRequest):
        asked.append(req)
        return "allow_always" if len(asked) == 1 else "deny"

    model = ScriptedModel([
        Turn("", [("run_command", {"command": "echo one"})]),
        Turn("", [("run_command", {"command": "echo two"})]),   # allowlisted by now
        Turn("", [("run_command", {"command": "printf three"})]),  # asks, denied
        Turn("ok"),
    ])
    h = Harness(model, ws, approver=approver, permission_level=PermissionLevel.ALLOWLIST)
    fins = [e for e in events(h, "go") if isinstance(e, ToolFinished)]
    assert [f.ok for f in fins] == [True, True, False]
    assert "one" in fins[0].output and "two" in fins[1].output
    assert "declined" in fins[2].output
    assert [r.detail for r in asked] == ["echo one", "printf three"]
    assert h.permissions.allowlist == ["echo"]
    assert Permissions.load(h.config_path).allowlist == ["echo"]  # persisted


def test_shell_disabled_at_level_none(ws: Path) -> None:
    model = ScriptedModel([Turn("", [("run_command", {"command": "echo hi"})]), Turn("ok")])
    h = Harness(model, ws, permission_level=PermissionLevel.NONE)
    fin = next(e for e in events(h, "go") if isinstance(e, ToolFinished))
    assert not fin.ok and "not available" in fin.output


def test_attachments_are_inlined(ws: Path) -> None:
    model = ScriptedModel([Turn("ok")])
    h = Harness(model, ws)
    list(h.run_turn("explain @README.md", ["README.md"]))
    user = h.agent.messages[1].content
    assert "<attached file README.md>" in user and "# Scratch" in user


def test_cancel_leaves_history_consistent(ws: Path) -> None:
    model = ScriptedModel([
        Turn("thinking about it", [("read_file", {"path": "README.md"}), ("read_file", {"path": "src/calc.py"})]),
        Turn("never reached"),
    ])
    h = Harness(model, ws)
    gen = h.run_turn("go")
    seen = []
    for ev in gen:
        seen.append(ev)
        if isinstance(ev, ToolFinished):      # cancel after the first tool ran
            h.cancel()
    assert isinstance(seen[-1], TurnEnded) and seen[-1].cancelled
    tool_msgs = [m for m in h.agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 2 and tool_msgs[1].content == "cancelled by user"
    # a following turn works
    model._script = [Turn("fine")]
    model._i = 0
    assert events(h, "and now?")[-1].text == "fine"


def test_abort_repairs_dangling_calls(ws: Path) -> None:
    model = ScriptedModel([
        Turn("", [("read_file", {"path": "README.md"}), ("read_file", {"path": "src/calc.py"})]),
        Turn("x"),
    ])
    h = Harness(model, ws)
    gen = h.run_turn("go")
    next(e for e in gen if isinstance(e, ToolFinished))   # first result is in; second call is pending
    gen.close()          # simulates a KeyboardInterrupt escaping the frontend loop
    h.abort()
    tool_msgs = [m for m in h.agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 2 and tool_msgs[-1].content == "cancelled by user"
    assert events(h, "still there?")[-1].text == "x"


# --- plan mode ----------------------------------------------------------------

def test_plan_mode_writes_file_and_build_switches(ws: Path) -> None:
    plan = "Goal: do the thing.\n\n- [ ] edit src/calc.py\n- [ ] add tests\n"
    model = ScriptedModel([
        Turn("", [("list_dir", {"path": "."})]),
        Turn("", [("write_plan", {"title": "Do the thing", "content": plan})]),
        Turn("Plan written."),
    ])
    h = Harness(model, ws, mode=Mode.PLAN)
    evs = events(h, "plan the thing")
    pu = next(e for e in evs if isinstance(e, PlanUpdated))
    assert pu.path == ".harness/plan.md" and pu.markdown.startswith("# Do the thing")
    assert h.plan_path.read_text() == pu.markdown
    assert h.changes() == []                       # the plan isn't a workspace change
    prompt = h.build_prompt()
    assert prompt and "- [ ] edit src/calc.py" in prompt
    assert h.mode is Mode.AGENT and "A plan exists" in h.system_prompt()
    archived = h.clear_plan()
    assert archived and archived.startswith(".harness/plans/") and not h.plan_path.exists()
    assert "A plan exists" not in h.system_prompt()


def test_write_plan_unavailable_in_ask_mode(ws: Path) -> None:
    model = ScriptedModel([Turn("", [("write_plan", {"title": "t", "content": "c"})]), Turn("ok")])
    h = Harness(model, ws, mode=Mode.ASK)
    fin = next(e for e in events(h, "go") if isinstance(e, ToolFinished))
    assert not fin.ok and not h.plan_path.exists()


# --- read tools ---------------------------------------------------------------

def test_read_tools(ws: Path) -> None:
    model = ScriptedModel([
        Turn("", [("read_file", {"path": "src/calc.py", "start_line": 2, "end_line": 2})]),
        Turn("", [("grep", {"pattern": "return", "glob": "*.py"})]),
        Turn("", [("find_files", {"glob": "*.md"})]),
        Turn("", [("read_file", {"path": "nope.py"})]),
        Turn("ok"),
    ])
    h = Harness(model, ws, mode=Mode.ASK)
    fins = [e for e in events(h, "go") if isinstance(e, ToolFinished)]
    assert fins[0].output == "src/calc.py lines 2-2 of 2\n2|     return a + b"
    assert fins[1].output == "src/calc.py:2: return a + b"
    assert fins[2].output == "README.md"
    assert not fins[3].ok and "no such file" in fins[3].output


def test_git_sees_edits(ws: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=ws, check=True)
    model = ScriptedModel([Turn("", [("edit_file", {"path": "README.md", "old": "hello", "new": "goodbye"})]), Turn("ok")])
    events(Harness(model, ws), "go")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True).stdout
    assert status.strip() == "M README.md"   # this is what Cursor's source-control view shows; .harness/ is git-ignored


# --- web tools (parsing only; no network) ---------------------------------------

DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=abc">Example <b>Docs</b></a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">The official docs.</a>
</div>
<div class="result">
  <a class="result__a" href="https://other.org/page">Other page</a>
  <a class="result__snippet">Second snippet</a>
</div>
"""


def test_search_html_parsing() -> None:
    from examples.harness.core import web
    results = web.parse_search_html(DDG_HTML)
    assert [(r.title, r.url, r.snippet) for r in results] == [
        ("Example Docs", "https://example.com/docs", "The official docs."),
        ("Other page", "https://other.org/page", "Second snippet"),
    ]


def test_html_to_text() -> None:
    from examples.harness.core import web
    page = "<html><head><title>T</title><style>x{}</style></head><body><nav>skip</nav><h1>Hi</h1><p>Para &amp; more</p><ul><li>one</li><li>two</li></ul><pre>a  b</pre><script>bad()</script></body></html>"
    title, text = web.html_to_text(page)
    assert title == "T"
    assert "skip" not in text and "bad()" not in text and "x{}" not in text
    assert text.startswith("# Hi\n\nPara & more\n\n- one\n- two")
    assert "a  b" in text


def test_web_tools_follow_permission(ws: Path, monkeypatch) -> None:
    from examples.harness.core import web
    monkeypatch.setattr(web, "search", lambda q, max_results=8: [web.SearchResult("t", "https://x", "s")])
    monkeypatch.setattr(web, "fetch", lambda url, max_chars=0: ("Title", "body text"))
    model = ScriptedModel([
        Turn("", [("web_search", {"query": "toki"})]),
        Turn("", [("fetch_url", {"url": "https://x"})]),
        Turn("ok"),
    ])
    h = Harness(model, ws, mode=Mode.ASK)
    assert {"web_search", "fetch_url"} <= {t.name for t in h.tools}
    fins = [e for e in events(h, "go") if isinstance(e, ToolFinished)]
    assert fins[0].ok and "https://x" in fins[0].output
    assert fins[1].ok and fins[1].output.startswith("Title\nhttps://x\n\nbody text")
    h.set_web(False)
    assert not any(t.kind == "web" for t in h.tools)
    assert "WEB:" not in h.system_prompt()
    assert Permissions.load(h.config_path).web is False


# --- sessions -------------------------------------------------------------------

def test_session_save_and_resume(ws: Path) -> None:
    model = ScriptedModel([Turn("", [("read_file", {"path": "README.md"})]), Turn("The README says hello.")])
    h = Harness(model, ws, mode=Mode.ASK)
    assert h.list_sessions() == []
    events(h, "what does the README say?")
    path = h.session_path
    assert path is not None and path.parent == ws / ".harness" / "sessions"
    sessions = h.list_sessions()
    assert len(sessions) == 1 and sessions[0].title == "what does the README say?" and sessions[0].turns == 1

    h2 = Harness(ScriptedModel([Turn("still here")]), ws)      # fresh process, default agent mode
    msgs = h2.resume_session(path)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert h2.mode is Mode.ASK and h2.turns == 1 and h2.usage.total_tokens > 0
    assert h2.agent.messages[0].role == "system" and "MODE: ask" in h2.agent.messages[0].content
    assert h2.agent.messages[1:] == msgs
    assert events(h2, "anything?")[-1].text == "still here"
    assert h2.session_path == path and h2.list_sessions()[0].turns == 2   # continued in the same file

    h2.reset()
    assert h2.session_path is None
    events(h2, "new topic")
    assert len(h2.list_sessions()) == 2


def test_session_title_ignores_attachments(ws: Path) -> None:
    from examples.harness.core.sessions import title_from
    from toki import TokiMessage
    m = TokiMessage(role="user", content="explain this\n\n<attached file README.md>\n# Scratch\n</attached>")
    assert title_from([m]) == "explain this"


# --- undo / redo ------------------------------------------------------------------

def _two_turn_harness(ws: Path) -> Harness:
    model = ScriptedModel([
        Turn("", [("edit_file", {"path": "src/calc.py", "old": "a + b", "new": "a * b"})]),
        Turn("", [("run_command", {"command": "echo hi"})]),
        Turn("edited"),
        Turn("", [("create_file", {"path": "b.txt", "content": "new\n"})]),
        Turn("", [("edit_file", {"path": "b.txt", "old": "new", "new": "newer"})]),
        Turn("created"),
    ])
    h = Harness(model, ws, permission_level=PermissionLevel.OPEN)
    events(h, "multiply instead")
    events(h, "make b")
    return h


def test_undo_reverts_files_and_conversation(ws: Path) -> None:
    h = _two_turn_harness(ws)
    hist = h.history()
    assert [(t.number, t.text, t.files, t.ran_shell) for t in hist] == [
        (1, "multiply instead", ["src/calc.py"], True), (2, "make b", ["b.txt"], False)]
    pv = h.undo_preview(1)
    assert pv and pv.files == ["b.txt"] and not pv.ran_shell
    r = h.undo(1)
    assert r and r.restored == ["b.txt"] and not (ws / "b.txt").exists()
    assert [m.role for m in h.agent.messages] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert h.turns == 1 and len(h.history()) == 1 and h.can_redo
    # first-touch checkpoint wins even though the turn edited b.txt twice
    r2 = h.redo()
    assert r2 and (ws / "b.txt").read_text() == "newer\n" and len(h.history()) == 2 and not h.can_redo
    # keep files: conversation rewinds, disk untouched
    r3 = h.undo(2, revert_files=False)
    assert r3 and r3.kept == ["b.txt", "src/calc.py"] and r3.restored == [] and r3.ran_shell
    assert (ws / "src/calc.py").read_text().endswith("a * b\n") and (ws / "b.txt").exists()
    assert [m.role for m in h.agent.messages] == ["system"] and h.turns == 0
    assert h.undo(1) is None


def test_undo_then_new_turn_clears_redo_and_saves(ws: Path) -> None:
    h = _two_turn_harness(ws)
    h.undo(1)
    h.model._script = [Turn("fresh")]; h.model._i = 0
    events(h, "something else")
    assert not h.can_redo and [t.text for t in h.history()] == ["multiply instead", "something else"]
    saved = h.sessions.load(h.session_path)
    assert [m.content for m in saved["messages"] if m.role == "user"][-1] == "something else"


def test_undo_of_resumed_turns_is_conversation_only(ws: Path) -> None:
    h = _two_turn_harness(ws)
    h2 = Harness(ScriptedModel([]), ws)
    h2.resume_session(h.session_path)
    hist = h2.history()
    assert [t.text for t in hist] == ["multiply instead", "make b"] and all(t.resumed for t in hist)
    pv = h2.undo_preview(1)
    assert pv and pv.resumed and pv.files == []
    r = h2.undo(1)
    assert r and r.restored == [] and (ws / "b.txt").exists()
    assert [m.role for m in h2.agent.messages][-1] == "assistant" and len(h2.history()) == 1
