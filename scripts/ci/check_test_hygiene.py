#!/usr/bin/env python3
"""Fail if tests reintroduce call-order spying — asserting *which* internal method
was called rather than the observable behavior.

The behavioral oracle (docs/test-plan.md) forbids "mocking internals only to prove
call order." This guard keeps that rule executable so the pattern cannot silently
creep back in. It is intentionally narrow (low false-positive): it flags assertions
on `*_called` spy flags and unittest-mock call-order assertions under tests/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"

# `assert store.search_called is True` / `assert x.foo_called is False`
SPY_FLAG = re.compile(r"^\s*assert\s+.*_called\s+is\s+(?:True|False)\b")
# `mock.assert_called*`, `.assert_awaited*`, `assert <x>.call_count == ...`
MOCK_CALLORDER = re.compile(r"\.assert_(?:called|awaited)\w*\(|\bcall_count\b")

failures: list[str] = []
for path in sorted(TESTS_DIR.rglob("test_*.py")):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if SPY_FLAG.match(line) or MOCK_CALLORDER.search(line):
            rel = path.relative_to(ROOT)
            failures.append(f"{rel}:{line_no}: call-order spy — assert behavior, not which method was called")

if failures:
    print("Test-hygiene check failed (call-order spying is forbidden by test-plan.md):")
    print("\n".join(failures))
    print("\nFix: assert the observable output of the behavior, not that an internal method ran.")
    sys.exit(1)

print("Test-hygiene check passed.")
