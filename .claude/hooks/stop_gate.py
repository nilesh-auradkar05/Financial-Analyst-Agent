#!/usr/bin/env python3
"""H12 — Stop: the turn may not end with a dirty tree, failing unit tests, or failing doc-sync.

Loop guard: exits immediately when stop_hook_active is set. Scope of the test run is limited to
tests/unit -x and only when app/ or evaluation/ changed, to keep turn latency bounded.
"""
from __future__ import annotations

import os
from pathlib import Path

from _common import REPO, block_turn, git, read_event, sh

ev = read_event()
if ev.get("stop_hook_active"):
    raise SystemExit(0)
if os.environ.get("SKIP_STOP_GATE") == "1":
    raise SystemExit(0)

problems: list[str] = []
dirty = git("status", "--porcelain").strip()
if dirty:
    files = [l[3:] for l in dirty.splitlines()]
    problems.append("working tree has uncommitted changes:\n    " + "\n    ".join(files[:15])
                    + "\n  → commit with the task's verification evidence, or state explicitly why it stays uncommitted.")

code_changed = any(f.startswith(("app/", "evaluation/", "tests/")) for f in
                   (git("diff", "--name-only", "HEAD~1..HEAD") + dirty).split())
if code_changed and (REPO / "tests" / "unit").exists():
    r = sh("uv run pytest tests/unit -q -x -p no:cacheprovider 2>&1 | tail -15", timeout=270)
    if r.returncode != 0:
        problems.append("unit tests failing:\n" + r.stdout.strip())

for script in ("scripts/ci/check_doc_sync.py", "scripts/ci/check_sprint_map.py"):
    if (REPO / script).exists() and Path(REPO / "docs").exists():
        r = sh(f"uv run python {script}", timeout=60)
        if r.returncode != 0:
            problems.append(f"{script} failed:\n" + (r.stdout + r.stderr).strip()[-600:])

if problems:
    block_turn("Done conditions not met (SPEC verification discipline):\n- " + "\n- ".join(problems))
raise SystemExit(0)
