#!/usr/bin/env python3
"""H13 — quality gate: ruff + mypy when a turn stops; ruff + mypy + pytest before every `git commit`.

PreToolUse Bash: only commands that run `git commit` are gated; each failing check is named in the
deny reason. A `test(<id>): red` commit (CLAUDE.md §3 step 4) may change Python files only under
tests/; those files are excluded from mypy and pytest, while ruff still covers them.
Stop: after a code change, ruff and mypy must pass (H12 runs the unit tests).
Git-ignored paths are excluded from mypy, because CI never sees them. Checks run in the event's
`cwd` when given (a commit may happen in a worktree), else in the project directory.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from _common import REPO, block_turn, deny, read_event

COMMIT = re.compile(r"(?:^|[;&|(]\s*|\n\s*)git(?:\s+-[cC]\s+\S+)*\s+commit\b")
RED = re.compile(r"test\([^)\s]+\): red\b")
CODE_DIRS = ("app/", "evaluation/", "dataops/", "scripts/", "tests/", ".claude/hooks/")

ev = read_event()
root = Path(ev.get("cwd") or REPO)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=30).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def changed_paths() -> list[str]:
    status = git("status", "--porcelain", "--untracked-files=all")
    return [line[3:].split(" -> ")[-1] for line in status.splitlines() if line.strip()]


def mypy_command(excluded: list[str]) -> str:
    ignored = git("ls-files", "--others", "--ignored", "--exclude-standard", "--directory").splitlines()
    patterns = [f"^{re.escape(p)}" for p in ignored if p.endswith(("/", ".py"))]
    patterns += [f"^{re.escape(p)}$" for p in excluded]
    return "uv run mypy . " + " ".join(shlex.quote(f"--exclude={p}") for p in patterns)


def failure(label: str, cmd: str, timeout: int) -> str | None:
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"{label} timed out after {timeout}s"
    if r.returncode == 0:
        return None
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
    return f"{label} failed:\n{tail}"


if ev.get("hook_event_name") == "Stop":
    if ev.get("stop_hook_active") or os.environ.get("SKIP_STOP_GATE") == "1":
        raise SystemExit(0)
    touched = git("diff", "--name-only", "HEAD~1..HEAD").split() + changed_paths()
    if not any(p.startswith(CODE_DIRS) for p in touched):
        raise SystemExit(0)
    found = [f for f in (failure("ruff", "uv run ruff check .", 120),
                         failure("mypy", mypy_command([]), 300)) if f]
    if found:
        block_turn("H13 quality gate: ruff and mypy must pass before the task is done.\n\n"
                   + "\n\n".join(found))
    raise SystemExit(0)

command = (ev.get("tool_input") or {}).get("command") or ""
if ev.get("tool_name") != "Bash" or not COMMIT.search(command):
    raise SystemExit(0)

excluded: list[str] = []
if RED.search(command):
    changed = [p for p in changed_paths() if p.endswith(".py")]
    outside = [p for p in changed if not p.startswith("tests/")]
    if outside:
        deny("H13: a red commit (`test(<id>): red`) may change Python files only under tests/; "
             "found: " + ", ".join(outside))
    excluded = changed

pytest_cmd = "uv run pytest -q -p no:cacheprovider " + " ".join(
    shlex.quote(f"--ignore={p}") for p in excluded)
found = [f for f in (failure("ruff", "uv run ruff check .", 120),
                     failure("mypy", mypy_command(excluded), 300),
                     failure("pytest", pytest_cmd, 600)) if f]
if found:
    deny("H13 quality gate: commit blocked until ruff, mypy and pytest all pass.\n\n"
         + "\n\n".join(found))
raise SystemExit(0)
