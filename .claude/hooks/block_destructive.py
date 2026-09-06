#!/usr/bin/env python3
"""H9 — PreToolUse Bash: block irreversible git/filesystem/store operations."""
from __future__ import annotations

import re

from _common import deny, read_event

RULES = [
    (re.compile(r"git\s+push\b.*(--force\b|-f\b|--force-with-lease)"), "force push"),
    (re.compile(r"git\s+reset\s+--hard"), "git reset --hard"),
    (re.compile(r"git\s+(clean\s+-[a-z]*f|branch\s+-D|checkout\s+--\s+\.)"), "destructive git"),
    (re.compile(r"git\s+commit\b.*--no-verify"), "bypassing commit hooks"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(?!/tmp/|\./?tmp/|/home/claude/)(/|\.|~|\*|\$)"), "recursive delete outside /tmp"),
    (re.compile(r"delete_collection|drop_collection|DROP\s+TABLE|FLUSHALL|FLUSHDB", re.I), "store/collection deletion"),
    (re.compile(r"\bpip\s+install\b(?!.*--break-system-packages)|\buv\s+pip\s+install\b"), "ad-hoc install; use `uv add` so uv.lock stays authoritative"),
]

ev = read_event()
cmd = (ev.get("tool_input", {}) or {}).get("command", "") or ""
for pat, label in RULES:
    if pat.search(cmd):
        deny(f"Blocked ({label}). If genuinely required, run it manually outside the agent and record why in tasks/todo.md.")
raise SystemExit(0)
