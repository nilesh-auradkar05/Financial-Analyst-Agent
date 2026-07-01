#!/usr/bin/env python3
"""Fail if obvious residue from the PulsePress/AWS/Terraform template remains in core Alpha docs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    ROOT / "CLAUDE.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "SPEC.md",
    ROOT / "docs" / "test-plan.md",
    ROOT / "docs" / "retrieval-benchmark.md",
]
FILES.extend(sorted((ROOT / "docs" / "adr").glob("*.md")))

PATTERNS = {
    "PulsePress": re.compile(r"\bPulsePress\b", re.IGNORECASE),
    "Terraform residue": re.compile(r"\bterraform\b", re.IGNORECASE),
}

failures: list[str] = []
for path in FILES:
    if not path.exists():
        failures.append(f"missing file: {path.relative_to(ROOT)}")
        continue
    text = path.read_text(encoding="utf-8")
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            failures.append(f"{path.relative_to(ROOT)}:{line_no}: {label}")

if failures:
    print("Scope-residue check failed:")
    print("\n".join(failures))
    sys.exit(1)

print("Scope-residue check passed.")
