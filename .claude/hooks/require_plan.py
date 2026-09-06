#!/usr/bin/env python3
"""H2 — UserPromptSubmit: implementation prompts require an unchecked plan item in tasks/todo.md.

Blocks only when the prompt clearly asks for implementation AND no open plan item exists.
Explanatory / review / planning prompts always pass.
"""
from __future__ import annotations

import re

from _common import REPO, block_turn, read_event

IMPLEMENT = re.compile(r"\b(implement|build|add|create|write the code|refactor|migrate|wire up|ship)\b", re.I)
EXEMPT = re.compile(r"\b(plan|explain|review|why|what|how does|design|propose|audit|todo)\b", re.I)

ev = read_event()
prompt = ev.get("prompt", "") or ""
if not IMPLEMENT.search(prompt) or EXEMPT.search(prompt):
    raise SystemExit(0)
todo = REPO / "tasks" / "todo.md"
open_items = [] if not todo.exists() else re.findall(r"^- \[ \] .+", todo.read_text(), re.M)
if open_items:
    raise SystemExit(0)
block_turn(
    "Governance (CLAUDE.md task loop): this looks like an implementation request but tasks/todo.md has no "
    "unchecked plan item. Write the plan first (checkable items + verification step), then re-send."
)
