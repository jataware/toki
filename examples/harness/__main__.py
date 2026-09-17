"""
Entry point: `python -m examples.harness [workspace] [--model provider:name] [--mode agent|ask|plan] ...`

A Cursor-style coding agent in the terminal, built on toki. See README.md in
this folder for a tour. Extra dependencies: rich, prompt_toolkit, plus a toki
backend (e.g. `pip install 'toki[openrouter]'`).
"""

from __future__ import annotations

import argparse
import sys

from .core import Mode, PermissionLevel, default_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m examples.harness",
        description="Cursor-style coding agent on toki. Type /help inside for commands.",
    )
    parser.add_argument("workspace", nargs="?", default=".", help="directory the agent may read and edit (default: cwd)")
    parser.add_argument("--model", "-m", default=default_spec(),
                        help="provider:name, e.g. openrouter:anthropic/claude-sonnet-4.5, anthropic:claude-sonnet-4-5, "
                             "ollama:qwen3:8b, or demo (no key needed). Default from $HARNESS_MODEL.")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.AGENT.value, help="starting mode")
    parser.add_argument("--permissions", "-p", choices=[p.value for p in PermissionLevel], default=None,
                        help="shell permission level (default: last saved in .harness/config.json, else ask)")
    parser.add_argument("--think", action="store_true", help="show the model's reasoning when the backend exposes it")
    parser.add_argument("--tui", action="store_true", help="use the Textual interface (sidebar, clickable rewind, diff cards) instead of the plain CLI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--new", action="store_true", help="start a new conversation without offering to resume")
    group.add_argument("--resume", nargs="?", const="last", metavar="N",
                       help="resume the most recent conversation, or the Nth most recent, without asking")
    args = parser.parse_args(argv)

    if args.tui:
        from .tui.app import run
    else:
        from .cli.app import run
    return run(
        args.workspace, args.model,
        mode=Mode(args.mode),
        permission_level=PermissionLevel(args.permissions) if args.permissions else None,
        thinking=args.think,
        resume="new" if args.new else args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
