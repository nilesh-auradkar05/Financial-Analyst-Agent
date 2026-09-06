"""Shared helpers for Alpha Analyst governance hooks (SPEC §14).

Protocol: event JSON on stdin; decisions on stdout as JSON; exit 0 unless noted.
Every hook must be safe to run outside Claude Code (manual smoke test):
    echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | .claude/hooks/<hook>.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()


def read_event() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def rel(path: str) -> str:
    """Repo-relative POSIX path; paths outside the repo are returned absolute."""
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def pre_tool_decision(decision: str, reason: str, *, updated_input: dict | None = None,
                      context: str | None = None) -> None:
    out: dict = {"hookEventName": "PreToolUse", "permissionDecision": decision,
                 "permissionDecisionReason": reason}
    if updated_input is not None:
        out["updatedInput"] = updated_input
    if context:
        out["additionalContext"] = context
    print(json.dumps({"hookSpecificOutput": out}))
    sys.exit(0)


def deny(reason: str) -> None:
    pre_tool_decision("deny", reason)


def block_turn(reason: str) -> None:
    """For UserPromptSubmit / Stop: block with reason."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def feedback(msg: str) -> None:
    """PostToolUse: surface a message to the model without blocking (exit 2 = stderr to model)."""
    sys.stderr.write(msg.rstrip() + "\n")
    sys.exit(2)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=15).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def sh(cmd: str, timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=timeout)
