"""
The tool set the model works through.

Every tool is a plain function taking a `ToolContext` plus JSON arguments and
returning a `ToolResult`. Keeping the model on this narrow, application-owned
surface (rather than a general shell) is what makes the permission story
simple: only `run_command` ever needs approval.

Tool kinds decide which modes see them:

    read   list_dir, read_file, grep, find_files   (all modes)
    write  edit_file, create_file, delete_file     (agent mode)
    plan   write_plan                              (plan and agent modes)
    shell  run_command                             (agent mode, if permitted)
    web    web_search, fetch_url                   (all modes, unless web is off)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from toki import StreamingToolSchema, ToolSchema

from .events import Event, FileChanged, PlanUpdated
from .permissions import Approver, PermissionRequest, Permissions
from .workspace import Workspace, WorkspaceError, truncate, unified_diff
from . import web

ToolKind = Literal["read", "write", "plan", "shell", "web"]

MAX_READ_CHARS = 40_000
MAX_GREP_RESULTS = 60
PLAN_FILE = "plan.md"


@dataclass
class ToolContext:
    workspace: Workspace
    permissions: Permissions
    approver: Approver

    @property
    def plan_path(self) -> Path:
        return self.workspace.harness_dir / PLAN_FILE


@dataclass
class ToolResult:
    output: str                      # returned to the model
    summary: str                     # one line for humans
    ok: bool = True
    events: list[Event] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., ToolResult]
    kind: ToolKind
    streaming: bool = False

    def schema(self) -> ToolSchema | StreamingToolSchema:
        wire = {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
        return StreamingToolSchema(wire) if self.streaming else ToolSchema(wire)

    def __call__(self, ctx: ToolContext, **args) -> ToolResult:
        try:
            return self.fn(ctx, **args)
        except WorkspaceError as e:
            return ToolResult(output=f"error: {e}", summary=str(e), ok=False)
        except TypeError as e:  # bad/missing arguments from the model
            return ToolResult(output=f"error: bad arguments for {self.name}: {e}", summary="bad arguments", ok=False)


def _params(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


# --- read tools -------------------------------------------------------------

def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> ToolResult:
    depth = max(1, min(int(depth), 6))
    out = ctx.workspace.tree(path, depth=depth)
    n = out.count("\n")
    return ToolResult(out, f"{path}  ({n} entries, depth {depth})")


def read_file(ctx: ToolContext, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
    text = ctx.workspace.read(path)
    lines = text.splitlines()
    total = len(lines)
    lo = max(1, int(start_line or 1))
    hi = min(total, int(end_line or total))
    if lo > total and total > 0:
        return ToolResult(f"error: {path} has only {total} lines", f"{path}: start_line past end", ok=False)
    width = len(str(hi)) if hi else 1
    body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(lo, hi + 1))
    if total == 0:
        body = "(empty file)"
    if len(body) > MAX_READ_CHARS:
        body = truncate(body, MAX_READ_CHARS) + "\n(use start_line/end_line to read a smaller range)"
    header = f"{path} lines {lo}-{hi} of {total}\n" if (lo != 1 or hi != total) else ""
    span = f"lines {lo}-{hi} of {total}" if (lo != 1 or hi != total) else f"{total} lines"
    return ToolResult(header + body, f"{path}  ({span})")


def grep(ctx: ToolContext, pattern: str, path: str = ".", glob: str | None = None, max_results: int = MAX_GREP_RESULTS) -> ToolResult:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(f"error: invalid regex: {e}", "invalid regex", ok=False)
    max_results = max(1, min(int(max_results), 500))
    hits: list[str] = []
    files_hit = 0
    for f in ctx.workspace.iter_files(path, glob=glob):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = ctx.workspace.relpath(f)
        had = False
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                had = True
                if len(hits) >= max_results:
                    break
        files_hit += had
        if len(hits) >= max_results:
            hits.append(f"... stopped at {max_results} matches; narrow the pattern, path, or glob")
            break
    if not hits:
        return ToolResult(f"no matches for /{pattern}/ under {path}" + (f" ({glob})" if glob else ""), f"/{pattern}/  no matches")
    return ToolResult("\n".join(hits), f"/{pattern}/  {min(len(hits), max_results)} matches in {files_hit} files")


def find_files(ctx: ToolContext, glob: str, path: str = ".", max_results: int = 200) -> ToolResult:
    files = ctx.workspace.iter_files(path, glob=glob)
    rels = [ctx.workspace.relpath(f) for f in files[: int(max_results)]]
    if not rels:
        return ToolResult(f"no files match {glob} under {path}", f"{glob}  no matches")
    more = f"\n... {len(files) - len(rels)} more" if len(files) > len(rels) else ""
    return ToolResult("\n".join(rels) + more, f"{glob}  {len(files)} files")


# --- write tools ------------------------------------------------------------

def edit_file(ctx: ToolContext, path: str, old: str, new: str) -> ToolResult:
    ws = ctx.workspace
    text = ws.read(path)
    if old == "":
        return ToolResult("error: `old` must not be empty; use create_file for new files", "empty `old`", ok=False)
    n = text.count(old)
    if n == 0:
        hint = _closest_hint(text, old)
        return ToolResult(f"error: `old` text not found in {path}.{hint}", f"{path}: old text not found", ok=False)
    if n > 1:
        return ToolResult(f"error: `old` text occurs {n} times in {path}; include more surrounding lines to make it unique", f"{path}: ambiguous ({n} matches)", ok=False)
    updated = text.replace(old, new, 1)
    rel = ws.relpath(ws.resolve(path))
    ws.write(rel, updated)
    diff = unified_diff(text, updated, rel)
    plus = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    minus = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return ToolResult(
        f"edited {rel} (+{plus} -{minus})",
        f"{rel}  +{plus} -{minus}",
        events=[FileChanged(rel, "modified", diff)],
    )


def create_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    ws = ctx.workspace
    p = ws.resolve(path)
    rel = ws.relpath(p)
    before = ws.read(rel) if p.exists() else None
    kind = ws.write(rel, content)
    diff = unified_diff(before or "", content, rel, missing_before=before is None)
    n = len(content.splitlines())
    return ToolResult(f"{kind} {rel} ({n} lines)", f"{rel}  {kind}, {n} lines", events=[FileChanged(rel, kind, diff)])  # type: ignore[arg-type]


def delete_file(ctx: ToolContext, path: str) -> ToolResult:
    ws = ctx.workspace
    rel = ws.relpath(ws.resolve(path))
    before = ws.read(rel)
    ws.delete(rel)
    diff = unified_diff(before, "", rel, missing_after=True)
    return ToolResult(f"deleted {rel}", f"{rel}  deleted", events=[FileChanged(rel, "deleted", diff)])


# --- plan tool --------------------------------------------------------------

def write_plan(ctx: ToolContext, title: str, content: str) -> ToolResult:
    ws = ctx.workspace
    rel = ws.relpath(ctx.plan_path)
    markdown = content if content.lstrip().startswith("#") else f"# {title.strip()}\n\n{content.strip()}\n"
    if not markdown.endswith("\n"):
        markdown += "\n"
    ws.write(rel, markdown)
    steps = sum(1 for l in markdown.splitlines() if l.lstrip().startswith("- ["))
    return ToolResult(
        f"plan written to {rel} ({steps} steps). The user will review it; do not start implementing.",
        f"{rel}  {steps} steps",
        events=[PlanUpdated(rel, markdown)],
    )


# --- shell tool -------------------------------------------------------------

def run_command(ctx: ToolContext, command: str, timeout: int = 60) -> ToolResult:
    verdict = ctx.permissions.check(command)
    if verdict == "blocked":
        return ToolResult("error: shell access is disabled in this session", "shell disabled", ok=False)
    if verdict == "ask":
        decision = ctx.approver(PermissionRequest(tool="run_command", detail=command))
        if decision == "deny":
            return ToolResult("The user declined to run this command. Ask before trying an alternative.", "denied by user", ok=False)
        if decision == "allow_always":
            ctx.permissions.allow(_prefix_of(command))
    code, out = ctx.workspace.run(command, timeout=max(1, min(int(timeout), 600)))
    status = "ok" if code == 0 else f"exit {code}"
    body = out or "(no output)"
    return ToolResult(f"[{status}]\n{body}", f"{command}  → {status}", ok=(code == 0))


def _prefix_of(command: str) -> str:
    """`pytest tests/ -q` -> `pytest`; `git status` -> `git status` (keep a subcommand)."""
    words = command.strip().split()
    if not words:
        return command
    if len(words) > 1 and words[0] in {"git", "uv", "npm", "pnpm", "yarn", "cargo", "docker", "make", "poetry", "pip"} and not words[1].startswith("-"):
        return " ".join(words[:2])
    return words[0]


def _closest_hint(text: str, old: str) -> str:
    """Point the model at a near-miss so it can fix whitespace/indent problems."""
    first = old.strip().splitlines()[0].strip() if old.strip() else ""
    if not first:
        return ""
    for i, line in enumerate(text.splitlines(), 1):
        if first in line:
            return f" A similar line exists at line {i}; re-read the file and copy the exact text, including indentation."
    return " Re-read the file; the text may have changed."


# --- web tools ---------------------------------------------------------------

def web_search(ctx: ToolContext, query: str, max_results: int = 8) -> ToolResult:
    try:
        results = web.search(query, max_results=max(1, min(int(max_results), 20)))
    except web.WebError as e:
        return ToolResult(f"error: {e}", str(e), ok=False)
    if not results:
        return ToolResult(f"no results for: {query}", f"{query!r}  no results")
    lines = [f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)]
    return ToolResult("\n".join(lines), f"{query!r}  {len(results)} results")


def fetch_url(ctx: ToolContext, url: str) -> ToolResult:
    try:
        title, text = web.fetch(url)
    except web.WebError as e:
        return ToolResult(f"error: {e}", str(e), ok=False)
    header = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
    return ToolResult(header + text, f"{title or url}  ({len(text):,} chars)")


# --- registry ---------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "list_dir",
            "List the files and folders under a directory (depth-limited tree). Start here to learn the layout.",
            _params({
                "path": {"type": "string", "description": "Directory relative to the workspace root. Default '.'"},
                "depth": {"type": "integer", "description": "How many levels to show (1-6). Default 2."},
            }, []),
            list_dir, "read",
        ),
        Tool(
            "read_file",
            "Read a text file with line numbers. Use start_line/end_line for large files.",
            _params({
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "First line to read (1-based)."},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)."},
            }, ["path"]),
            read_file, "read",
        ),
        Tool(
            "grep",
            "Search file contents with a regular expression. Returns path:line: text matches.",
            _params({
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Directory or file to search. Default '.'"},
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
                "max_results": {"type": "integer", "description": "Cap on matches. Default 60."},
            }, ["pattern"]),
            grep, "read",
        ),
        Tool(
            "find_files",
            "Find files by name glob, e.g. '*.toml' or 'test_*.py'.",
            _params({
                "glob": {"type": "string", "description": "Glob matched against file names or workspace-relative paths."},
                "path": {"type": "string", "description": "Directory to search under. Default '.'"},
            }, ["glob"]),
            find_files, "read",
        ),
        Tool(
            "edit_file",
            "Replace one exact block of text in an existing file. `old` must match exactly once (copy it from read_file, "
            "including indentation). Prefer several small edits over one large one.",
            _params({
                "path": {"type": "string", "description": "File to edit."},
                "old": {"type": "string", "description": "Exact existing text to replace."},
                "new": {"type": "string", "description": "Replacement text."},
            }, ["path", "old", "new"]),
            edit_file, "write", streaming=True,
        ),
        Tool(
            "create_file",
            "Create a new file (or overwrite an existing one) with the given content.",
            _params({
                "path": {"type": "string", "description": "File to create."},
                "content": {"type": "string", "description": "Full file content."},
            }, ["path", "content"]),
            create_file, "write", streaming=True,
        ),
        Tool(
            "delete_file",
            "Delete a file from the workspace.",
            _params({"path": {"type": "string", "description": "File to delete."}}, ["path"]),
            delete_file, "write",
        ),
        Tool(
            "write_plan",
            "Write or replace the plan file. Use markdown with a short goal, then '- [ ] step' checkboxes, "
            "naming the files each step touches. Call this once your research is done.",
            _params({
                "title": {"type": "string", "description": "Short plan title."},
                "content": {"type": "string", "description": "Plan body in markdown."},
            }, ["title", "content"]),
            write_plan, "plan", streaming=True,
        ),
        Tool(
            "web_search",
            "Search the web (DuckDuckGo). Use for documentation, library APIs, error messages, or anything not in the "
            "workspace. Follow up with fetch_url to read a result.",
            _params({
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "How many results (1-20). Default 8."},
            }, ["query"]),
            web_search, "web",
        ),
        Tool(
            "fetch_url",
            "Fetch a web page and return its readable text (HTML is stripped). Long pages are truncated.",
            _params({"url": {"type": "string", "description": "Absolute http(s) URL."}}, ["url"]),
            fetch_url, "web",
        ),
        Tool(
            "run_command",
            "Run a shell command in the workspace (tests, builds, git, formatters). Use the file tools for reading "
            "and editing instead of cat/sed. Output is captured; the command may require user approval.",
            _params({
                "command": {"type": "string", "description": "The command line to run."},
                "timeout": {"type": "integer", "description": "Seconds before the command is killed. Default 60."},
            }, ["command"]),
            run_command, "shell",
        ),
    ]
}

READ_TOOLS = [t for t in TOOLS.values() if t.kind == "read"]
