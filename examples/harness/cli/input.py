"""
Line input with prompt_toolkit: history, multiline (Alt+Enter), tab completion
for `/commands` and `@files`, and a status toolbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from ..core import Workspace
from .commands import COMMANDS

MAX_FILE_COMPLETIONS = 40


class HarnessCompleter(Completer):
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._files: list[str] | None = None

    def refresh(self) -> None:
        self._files = None

    def files(self) -> list[str]:
        if self._files is None:
            paths = self.workspace.iter_files(".")
            self._files = [self.workspace.relpath(p) for p in paths[:5000]]
            dirs = sorted({str(Path(f).parent) + "/" for f in self._files if "/" in f})
            self._files = dirs + self._files
        return self._files

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # slash commands: only at the very start of the input
        if text.startswith("/") and " " not in text:
            for c in COMMANDS:
                if c.name.startswith(text[1:]):
                    yield Completion("/" + c.name, start_position=-len(text), display=f"/{c.name} {c.usage}".rstrip(), display_meta=c.short)
            return
        # first argument of a command with fixed choices
        if text.startswith("/") and text.count(" ") == 1:
            name, _, partial = text[1:].partition(" ")
            for c in COMMANDS:
                if c.name == name and c.choices:
                    for choice in c.choices:
                        if choice.startswith(partial):
                            yield Completion(choice, start_position=-len(partial))
                    return
            if name in ("diff", "revert", "open"):
                yield from self._file_completions(partial, at=False)
            return
        # @file mentions anywhere
        if word.startswith("@"):
            yield from self._file_completions(word[1:], at=True)

    def _file_completions(self, partial: str, *, at: bool) -> Iterable[Completion]:
        n = 0
        prefix = "@" if at else ""
        for f in self.files():
            if partial.lower() in f.lower():
                yield Completion(prefix + f, start_position=-(len(partial) + (1 if at else 0)), display=f)
                n += 1
                if n >= MAX_FILE_COMPLETIONS:
                    return


STYLE = Style.from_dict({
    "prompt.agent": "bold ansigreen",
    "prompt.ask": "bold ansiblue",
    "prompt.plan": "bold ansimagenta",
    "prompt.arrow": "bold",
    "bottom-toolbar": "noreverse bg:default fg:ansibrightblack",
    "bottom-toolbar.key": "fg:ansicyan",
    "completion-menu.completion": "bg:ansibrightblack fg:ansiwhite",
    "completion-menu.completion.current": "bg:ansicyan fg:ansiblack",
    "completion-menu.meta.completion": "bg:ansibrightblack fg:ansigray",
    "completion-menu.meta.completion.current": "bg:ansicyan fg:ansiblack",
})


def make_session(workspace: Workspace, toolbar: Callable[[], FormattedText]) -> tuple[PromptSession, HarnessCompleter]:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        buf = event.current_buffer
        state = buf.complete_state
        if state:
            # Enter with the menu open applies the highlighted (or only) completion instead of submitting
            chosen = state.current_completion or (state.completions[0] if len(state.completions) == 1 else None)
            buf.complete_state = None
            if chosen is not None:
                buf.apply_completion(chosen)
                return          # second Enter sends
        buf.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    completer = HarnessCompleter(workspace)
    history_path = workspace.harness_dir / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        completer=completer,
        complete_while_typing=True,
        multiline=True,
        key_bindings=kb,
        style=STYLE,
        bottom_toolbar=toolbar,
        prompt_continuation=lambda width, line_number, is_soft_wrap: " " * (width - 2) + "│ ",
        reserve_space_for_menu=6,
    )
    return session, completer
