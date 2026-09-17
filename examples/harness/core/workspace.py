"""
Workspace: a directory the agent may read and edit.

- every path the model supplies is resolved inside `root` (no escaping via
  `..` or symlinks)
- noisy directories (`.git`, `node_modules`, ...) are skipped by tree/grep
- the first time a file is touched in a session its original content is kept
  so `diff()` and `revert()` work even outside git

Review happens in your editor: with git, Cursor's Source Control view shows
each change and lets you keep or discard it per file or per hunk.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".idea", ".vscode", ".harness", ".DS_Store", ".eggs", "*.egg-info",
}
HARNESS_DIR = ".harness"


class WorkspaceError(Exception):
    """Raised for paths outside the workspace or other tool-level failures.
    The message is returned to the model as the tool result."""


@dataclass
class Snapshot:
    """Original content of a file before the session first touched it.
    `content is None` means the file did not exist."""
    content: str | None


@dataclass
class Workspace:
    root: Path
    _snapshots: dict[str, Snapshot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    # ----- paths -------------------------------------------------------------

    @property
    def harness_dir(self) -> Path:
        return self.root / HARNESS_DIR

    def ensure_harness_dir(self) -> Path:
        """Create `.harness/` with a self-ignoring .gitignore so its state never appears in git."""
        d = self.harness_dir
        d.mkdir(parents=True, exist_ok=True)
        gi = d / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n")
        return d

    def resolve(self, rel: str | os.PathLike[str]) -> Path:
        """Resolve `rel` inside the workspace or raise `WorkspaceError`."""
        raw = Path(os.path.expanduser(str(rel).strip() or "."))
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path is outside the workspace: {rel}")
        return resolved

    def relpath(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() if path != self.root else "."

    def is_ignored(self, path: Path) -> bool:
        return any(fnmatch.fnmatch(part, pat) for part in path.parts for pat in IGNORED_DIRS)

    def is_git_repo(self) -> bool:
        return (self.root / ".git").exists()

    # ----- listing ----------------------------------------------------------

    def tree(self, rel: str = ".", *, depth: int = 2, max_entries: int = 200) -> str:
        """Indented listing of `rel`, directories first, ignoring noise dirs."""
        start = self.resolve(rel)
        if not start.exists():
            raise WorkspaceError(f"no such path: {rel}")
        if start.is_file():
            return self.relpath(start)
        lines = [self.relpath(start) + "/"]
        count = 0
        truncated = False

        def walk(d: Path, level: int) -> None:
            nonlocal count, truncated
            if truncated:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return
            entries = [e for e in entries if not self.is_ignored(e.relative_to(self.root))]
            for e in entries:
                if count >= max_entries:
                    truncated = True
                    return
                count += 1
                indent = "  " * level
                if e.is_dir():
                    n = _count_children(e)
                    if level < depth:
                        lines.append(f"{indent}{e.name}/")
                        walk(e, level + 1)
                    else:
                        lines.append(f"{indent}{e.name}/  ({n} items)")
                else:
                    lines.append(f"{indent}{e.name}")

        walk(start, 1)
        if truncated:
            lines.append(f"... truncated at {max_entries} entries; list a subdirectory for more")
        return "\n".join(lines)

    def iter_files(self, rel: str = ".", *, glob: str | None = None) -> list[Path]:
        start = self.resolve(rel)
        out: list[Path] = []
        if start.is_file():
            return [start]
        for dirpath, dirnames, filenames in os.walk(start):
            d = Path(dirpath)
            dirnames[:] = sorted(n for n in dirnames if not self.is_ignored(d.joinpath(n).relative_to(self.root)))
            for name in sorted(filenames):
                p = d / name
                r = p.relative_to(self.root).as_posix()
                if self.is_ignored(Path(r)):
                    continue
                if glob and not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(r, glob)):
                    continue
                out.append(p)
        return out

    # ----- reading ----------------------------------------------------------

    def read(self, rel: str) -> str:
        p = self.resolve(rel)
        if not p.exists():
            raise WorkspaceError(f"no such file: {rel}")
        if p.is_dir():
            raise WorkspaceError(f"{rel} is a directory; use list_dir")
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise WorkspaceError(f"{rel} is not a UTF-8 text file ({p.stat().st_size} bytes)")

    # ----- writing (snapshotted) --------------------------------------------

    def _remember(self, rel: str, p: Path) -> None:
        if rel in self._snapshots:
            return
        if p.exists() and p.is_file():
            try:
                self._snapshots[rel] = Snapshot(p.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                self._snapshots[rel] = Snapshot(None)
        else:
            self._snapshots[rel] = Snapshot(None)

    def write(self, rel: str, content: str) -> str:
        """Write `content` to `rel`, creating parents. Returns 'created' | 'modified'."""
        p = self.resolve(rel)
        rel = self.relpath(p)
        if p.is_dir():
            raise WorkspaceError(f"{rel} is a directory")
        existed = p.exists()
        self._remember(rel, p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return "modified" if existed else "created"

    def delete(self, rel: str) -> None:
        p = self.resolve(rel)
        rel = self.relpath(p)
        if not p.exists():
            raise WorkspaceError(f"no such file: {rel}")
        if p.is_dir():
            raise WorkspaceError(f"{rel} is a directory; only files can be deleted")
        self._remember(rel, p)
        p.unlink()

    # ----- change tracking --------------------------------------------------

    def changed_files(self) -> list[tuple[str, str]]:
        """[(relpath, 'created'|'modified'|'deleted'|'unchanged')] for touched files."""
        out = []
        for rel, snap in self._snapshots.items():
            p = self.root / rel
            exists = p.exists()
            if snap.content is None and exists:
                kind = "created"
            elif snap.content is not None and not exists:
                kind = "deleted"
            elif snap.content is not None and exists and p.read_text(encoding="utf-8", errors="replace") != snap.content:
                kind = "modified"
            else:
                kind = "unchanged"
            out.append((rel, kind))
        return out

    def diff(self, rel: str) -> str:
        """Unified diff of `rel` against its session snapshot ('' if unchanged)."""
        rel = self.relpath(self.resolve(rel))
        snap = self._snapshots.get(rel)
        before = snap.content if snap else None
        p = self.root / rel
        after = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        if before is None and after is None:
            return ""
        return unified_diff(before or "", after or "", rel, missing_before=before is None, missing_after=after is None)

    def revert(self, rel: str) -> str:
        """Restore `rel` to its session snapshot. Returns a short description."""
        rel = self.relpath(self.resolve(rel))
        snap = self._snapshots.get(rel)
        if snap is None:
            raise WorkspaceError(f"{rel} was not changed this session")
        p = self.root / rel
        if snap.content is None:
            if p.exists():
                p.unlink()
            desc = f"removed {rel}"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(snap.content, encoding="utf-8")
            desc = f"restored {rel}"
        del self._snapshots[rel]
        return desc

    def forget_changes(self) -> None:
        self._snapshots.clear()

    # ----- shell ------------------------------------------------------------

    def run(self, command: str, *, timeout: int = 60, max_chars: int = 20_000) -> tuple[int, str]:
        """Run `command` with the workspace as cwd. Returns (exit_code, combined output)."""
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.root, capture_output=True, text=True,
                timeout=timeout, errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "")
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            return 124, truncate(partial, max_chars) + f"\n(timed out after {timeout}s)"
        out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode, truncate(out.strip(), max_chars)

    def has_executable(self, name: str) -> bool:
        return shutil.which(name) is not None


def unified_diff(before: str, after: str, rel: str, *, missing_before: bool = False, missing_after: bool = False) -> str:
    a = "/dev/null" if missing_before else f"a/{rel}"
    b = "/dev/null" if missing_after else f"b/{rel}"
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=a, tofile=b, n=3,
    )
    text = "".join(lines)
    # make sure the last line renders even without a trailing newline
    return text if text.endswith("\n") else text + "\n"


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n... ({len(text) - max_chars} chars omitted) ...\n{tail}"


def _count_children(d: Path) -> int:
    try:
        return sum(1 for _ in d.iterdir())
    except PermissionError:
        return 0
