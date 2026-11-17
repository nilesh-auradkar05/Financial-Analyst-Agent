#!/usr/bin/env python3
"""Fail if an eval result or registry record lacks a SPEC §11.2 lineage key.

Same pattern and rule as hook H5 (.claude/hooks/protect_paths.py): every JSON file under
`evaluation/<name>_res/` or `evaluation/registry/runs/` must contain each lineage key as a
quoted string. Legacy results live under `evaluation/archive/`, outside both patterns.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EVAL_RESULT = re.compile(r"^evaluation/([\w-]+_res|registry/runs)/.*\.json$")
LINEAGE = (
    "git_sha",
    "model",
    "temperature",
    "prompt_version",
    "evidence_release",
    "snapshot_hash",
    "fixture_version",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
root: Path = parser.parse_args().root

failures: list[str] = []
for path in sorted((root / "evaluation").rglob("*.json")):
    rel = path.relative_to(root).as_posix()
    if not EVAL_RESULT.match(rel):
        continue
    body = path.read_text(encoding="utf-8")
    missing = [key for key in LINEAGE if f'"{key}"' not in body]
    if missing:
        failures.append(f"{rel}: missing {missing}")

if failures:
    print("Eval-lineage check failed (SPEC §11.2):")
    print("\n".join(failures))
    sys.exit(1)

print("Eval-lineage check passed.")
