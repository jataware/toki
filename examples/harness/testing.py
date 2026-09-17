"""
A scripted toki model for tests and for `--model demo`.

`ScriptedModel` implements the four raw-I/O hooks of `toki.BaseModel`, so the
harness runs its real streaming path (per-character content, per-fragment tool
arguments) without a network. A script is either a list of `Turn`s consumed in
order, or a callable `(messages) -> Turn` that can react to tool results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterator, Sequence

from toki import BaseModel, TokenCountEstimate, TokiMessage, TokiToolCall, TokiToolFunction, TokiUsageMetadata
from toki.model import _RawContentChunk, _RawThoughtChunk, _RawToolCallChunk, _RawTurn, _RawUsage


@dataclass
class Turn:
    content: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    thought: str = ""


Script = Sequence[Turn] | Callable[[list[TokiMessage]], Turn]


class ScriptedModel(BaseModel):
    def __init__(self, script: Script, *, chunk_size: int = 5) -> None:
        super().__init__()
        self.model = "scripted"
        self._script = script
        self._i = 0
        self._n = 0  # monotonically increasing call counter for unique ids
        self._chunk = chunk_size
        self.calls: list[list[TokiMessage]] = []  # every prompt the model saw

    def _next(self, messages: list[TokiMessage]) -> Turn:
        self.calls.append(list(messages))
        self._n += 1
        if callable(self._script):
            return self._script(messages)
        if self._i >= len(self._script):
            return Turn("(script exhausted)")
        turn = self._script[self._i]
        self._i += 1
        return turn

    def _tool_calls(self, turn: Turn) -> list[TokiToolCall]:
        return [
            TokiToolCall(id=f"call_{self._n}_{k}", function=TokiToolFunction(name, args))
            for k, (name, args) in enumerate(turn.tool_calls)
        ]

    def _usage(self, messages: list[TokiMessage], turn: Turn) -> TokiUsageMetadata:
        p = sum(len(m.content) for m in messages) // 4
        c = len(turn.content) // 4 + sum(len(json.dumps(a)) for _, a in turn.tool_calls) // 4
        return TokiUsageMetadata(p, c, p + c)

    # ----- raw hooks ---------------------------------------------------------

    def _raw_blocking(self, messages, tools, *, capture_thinking, **kw) -> _RawTurn:
        turn = self._next(messages)
        return _RawTurn(turn.content, self._tool_calls(turn), turn.thought, usage=self._usage(messages, turn))

    def _raw_streaming(self, messages, tools, *, capture_thinking, **kw) -> Iterator:
        turn = self._next(messages)
        n = self._chunk
        if capture_thinking and turn.thought:
            for i in range(0, len(turn.thought), n):
                yield _RawThoughtChunk(turn.thought[i : i + n])
        for i in range(0, len(turn.content), n):
            yield _RawContentChunk(turn.content[i : i + n])
        for idx, (name, args) in enumerate(turn.tool_calls):
            yield _RawToolCallChunk(index=idx, id=f"call_{self._n}_{idx}", name=name)
            payload = json.dumps(args)
            for i in range(0, len(payload), n):
                yield _RawToolCallChunk(index=idx, arguments_fragment=payload[i : i + n])
        yield _RawUsage(self._usage(messages, turn))

    async def _raw_blocking_async(self, messages, tools, *, capture_thinking, **kw) -> _RawTurn:
        return self._raw_blocking(messages, tools, capture_thinking=capture_thinking, **kw)

    async def _raw_streaming_async(self, messages, tools, *, capture_thinking, **kw) -> AsyncIterator:
        for chunk in self._raw_streaming(messages, tools, capture_thinking=capture_thinking, **kw):
            yield chunk

    def count_tokens(self, messages, *, tools=None, kind="offline"):
        raw = sum(len(TokiMessage.from_dict(m).content) for m in messages) // 4
        return TokenCountEstimate(prompt_tokens=raw, raw_prompt_tokens=raw, safety_factor=1.0)


# --- the `--model demo` tour ---------------------------------------------------

DEMO_FILE = "HARNESS_DEMO.md"


def demo_model() -> ScriptedModel:
    """A canned session that exercises every event type, keyed off what the user says."""

    def script(messages: list[TokiMessage]) -> Turn:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        ask = (last_user.content if last_user else "").lower()
        last = messages[-1]
        system = messages[0].content if messages and messages[0].role == "system" else ""
        agent_mode = "MODE: agent" in system
        plan_mode = "MODE: plan" in system

        # follow-ups after a tool result
        if last.role == "tool":
            recent = [m for m in messages if m.role == "assistant" and m.tool_calls]
            done = [tc.function.name for m in recent for tc in m.tool_calls]
            if done[-1] == "list_dir":
                if plan_mode:
                    return Turn(tool_calls=[("write_plan", {
                        "title": "Add a demo file",
                        "content": "Goal: show how a plan is written to disk and then implemented.\n\n"
                                   f"- [ ] Create `{DEMO_FILE}` with a short greeting\n"
                                   "- [ ] Add a second line explaining the harness\n"
                                   "- [ ] Confirm the file exists with `ls`\n",
                    })])
                if agent_mode and ("demo" in ask or "implement the plan" in ask):
                    return Turn(
                        "I'll create a small file so you can see an edit land.",
                        [("create_file", {"path": DEMO_FILE, "content": "# Hello from the harness\n\nThis file was created by the demo model.\n"})],
                    )
                return Turn("Here's the layout. The tree above shows the top two levels of the workspace. "
                            "Ask me about any file and I'll read it, or say `demo` in agent mode to watch an edit.")
            if done[-1] == "write_plan":
                return Turn("I wrote a three-step plan to `.harness/plan.md`. Review it with `/plan`, edit it with `/plan edit`, "
                            "then run `/build` to implement it.")
            if done[-1] == "create_file":
                return Turn(
                    "Now a targeted edit: replacing the second paragraph.",
                    [("edit_file", {
                        "path": DEMO_FILE,
                        "old": "This file was created by the demo model.\n",
                        "new": "This file was created by the demo model.\n\nEdits stream in live and are reviewed as diffs in your editor.\n",
                    })],
                )
            if done[-1] == "edit_file":
                if "run_command" in system.lower() or "SHELL: run_command" in system:
                    return Turn("Let me confirm the file exists.", [("run_command", {"command": f"ls -la {DEMO_FILE}"})])
                return Turn(f"Done: created and edited `{DEMO_FILE}`. Use `/changes` to list it, `/diff` to see it, or `/revert` to undo.")
            if done[-1] == "run_command":
                return Turn(f"All done. `{DEMO_FILE}` was created and edited; use `/changes`, `/diff`, or `/revert {DEMO_FILE}` to review or undo it.")
            if done[-1] == "read_file":
                return Turn("That's the file. Notice the tool card above shows the path and line count; the full text went to the model, not the screen.")
            return Turn("Okay.")

        # fresh user message
        if any(w in ask for w in ("read", "show", "open")) and "readme" in ask:
            return Turn("Reading the README.", [("read_file", {"path": "README.md", "start_line": 1, "end_line": 40})])
        if ask.startswith("implement the plan"):
            return Turn("Starting on the plan.", [("list_dir", {"path": ".", "depth": 1})])
        if "demo" in ask or "layout" in ask or "structure" in ask or "what" in ask or plan_mode:
            return Turn("Let me look at the workspace first.", [("list_dir", {"path": ".", "depth": 2})], thought="The user wants an overview; list the tree.")
        return Turn(
            "This is the scripted demo model, so I only know a few tricks: ask `what's the layout?`, `read the README`, "
            "say `demo` in agent mode to watch an edit, or switch to `/plan` and ask for anything."
        )

    return ScriptedModel(script)
