#!/usr/bin/env python3
"""Check that SPEC and sprint-plan agree on the S0-S10 sprint map."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
sprint = (ROOT / "docs" / "sprint-plan.md").read_text(encoding="utf-8")

required = [f"S{i}" for i in range(11)]
missing: list[str] = []
for sid in required:
    if not re.search(rf"\b{sid}\b", spec):
        missing.append(f"SPEC missing {sid}")
    if not re.search(rf"\b{sid}\b", sprint):
        missing.append(f"sprint-plan missing {sid}")

if "S5–S10" not in sprint and "S5-S10" not in sprint:
    missing.append("sprint-plan header must describe horizon as S5-S10")

if missing:
    print("Sprint-map check failed:")
    print("\n".join(missing))
    sys.exit(1)

print("Sprint-map check passed.")
