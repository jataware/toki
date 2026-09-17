"""Headless test of the Textual frontend (Textual's `run_test` pilot). Skipped if textual isn't installed."""

from __future__ import annotations

from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from examples.harness.core import Harness, Mode, PermissionLevel  # noqa: E402
from examples.harness.testing import ScriptedModel, Turn  # noqa: E402
from examples.harness.tui.app import (  # noqa: E402
    AssistantMessage, ChangeItem, ChatInput, DiffCard, HarnessTUI, PermissionModal, PlanCard, ToolCard, UndoModal, UserMessage,
)
from textual.widgets import ListView, Markdown  # noqa: E402


async def _settle(app: HarnessTUI, pilot, *, until=None, tries: int = 200) -> None:
    for _ in range(tries):
        await pilot.pause(0.05)
        if (until() if until else not app._busy):
            return
    raise AssertionError("timed out waiting for the app")


@pytest.mark.asyncio
async def test_tui_full_turn_rewind_and_redo(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Scratch\n")
    model = ScriptedModel([
        Turn("Looking.", [("read_file", {"path": "README.md"})]),
        Turn("Adding a file.", [("create_file", {"path": "notes.md", "content": "# notes\n"})]),
        Turn("Checking.", [("run_command", {"command": "echo checked"})]),
        Turn("Done: created `notes.md`."),
    ])
    h = Harness(model, tmp_path, mode=Mode.AGENT, permission_level=PermissionLevel.ASK)
    app = HarnessTUI(h)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        inp = app.query_one("#input", ChatInput)
        inp.load_text("make notes")
        await pilot.press("enter")

        await _settle(app, pilot, until=lambda: isinstance(app.screen, PermissionModal))
        assert "echo checked" in app.screen.req.detail  # type: ignore[attr-defined]
        await pilot.press("y")
        await _settle(app, pilot)

        assert len(app.query(UserMessage)) == 1
        assert len(app.query(ToolCard)) == 3 and len(app.query(DiffCard)) == 1
        titles = [c.query_one("Collapsible").title for c in app.query(ToolCard)]
        assert titles[0].startswith("▢ read_file  README.md") and "→ ok" in titles[2]
        assert app.query(AssistantMessage).last().query_one(Markdown)._markdown.startswith("Done")
        assert (tmp_path / "notes.md").exists()
        await pilot.pause(0.2)
        assert [i.path for i in app.query_one("#changes", ListView).query(ChangeItem)] == ["notes.md"]

        # rewind button → undo modal → undo files
        app.query_one("UserMessage .rewind").press()
        await _settle(app, pilot, until=lambda: isinstance(app.screen, UndoModal))
        await pilot.press("u")
        await pilot.pause(0.3)
        assert len(h.history()) == 0 and len(app.query(UserMessage)) == 0
        assert not (tmp_path / "notes.md").exists()
        assert inp.text == "make notes"                 # message is back in the input box

        inp.clear(); inp.load_text("/redo"); await pilot.press("enter")
        await pilot.pause(0.3)
        assert len(h.history()) == 1 and len(app.query(UserMessage)) == 1 and len(app.query(ToolCard)) == 3
        assert (tmp_path / "notes.md").exists()


@pytest.mark.asyncio
async def test_tui_plan_mode_and_sidebar(tmp_path: Path) -> None:
    model = ScriptedModel([
        Turn("", [("write_plan", {"title": "Do it", "content": "- [ ] step one\n"})]),
        Turn("Plan written."),
    ])
    h = Harness(model, tmp_path, permission_level=PermissionLevel.NONE)
    app = HarnessTUI(h)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        inp = app.query_one("#input", ChatInput)
        inp.load_text("/plan"); await pilot.press("enter"); await pilot.pause(0.1)
        assert h.mode is Mode.PLAN
        inp.load_text("plan it"); await pilot.press("enter")
        await _settle(app, pilot)
        assert len(app.query(PlanCard)) == 1 and h.plan_path.exists()
        await pilot.pause(0.2)
        assert not app.query_one("#side-build").disabled
        # ask mode via command strips write tools; /web off flips the switch
        inp.load_text("/ask"); await pilot.press("enter"); await pilot.pause(0.1)
        assert h.mode is Mode.ASK
        inp.load_text("/web off"); await pilot.press("enter"); await pilot.pause(0.2)
        assert h.permissions.web is False and app.query_one("#web").value is False
