#!/usr/bin/env python3
"""Lightweight semantic sync check for CLAUDE.md and AGENTS.md.

This is not a theorem prover, because apparently we are still in Python, not a formal-methods monastery.
It checks that both files preserve the required operating-rule families.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "CLAUDE.md": (ROOT / "CLAUDE.md").read_text(encoding="utf-8"),
    "AGENTS.md": (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
}

REQUIRED_PHRASES = [
    "Trace",
    "Plan first",
    "behavior-first tests",
    "smallest necessary change",
    "Doc Sync Check",
    "record command results",
    "tasks/lessons.md",
    "Backend client objects never escape `rag/`",
    "Do not read or print `.env`",
    "single-axis rule",
    "Gold labels would be authored against unstable chunk IDs",
    "work is about to be marked done without recorded verification",
    "uv run ruff check .",
    "VECTOR_BACKEND=chroma",
    "VECTOR_BACKEND=qdrant",
    "correction, root cause, prevention rule",
]

failures: list[str] = []
for name, text in FILES.items():
    lowered = text.lower()
    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in lowered:
            failures.append(f"{name} missing phrase: {phrase}")

for name, text in FILES.items():
    lowered = text.lower()
    for heading in ["Task loop", "Hard stops", "Coding conventions", "Verification", "Correction handling"]:
        if heading.lower() not in lowered:
            failures.append(f"{name} missing heading concept: {heading}")

if failures:
    print("Doc-sync check failed:")
    print("\n".join(failures))
    sys.exit(1)

print("Doc-sync check passed.")
