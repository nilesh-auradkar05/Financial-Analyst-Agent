#!/usr/bin/env python3
"""H6 — PreToolUse Bash: refuse benchmark runs when uncommitted changes span more than one axis."""
from __future__ import annotations

from _common import deny, git, read_event

BENCH = ("quality_baseline", "retrieval_main", "run_rag_quality_eval", "run_shared_retrieval_benchmark",
         "run_retrieval_method_matrix", "latency_baseline")
AXES = {
    "prompt": ("app/prompts/", "app/services/llm.py"),
    "retrieval": ("app/components/retrieval/",),
    "llm": ("app/llm/", "app/config.py"),
    "graph": ("app/agents/",),
    "verifier": ("app/verification/", "evaluation/grounding.py", "evaluation/semantic_grounding.py"),
}

ev = read_event()
cmd = (ev.get("tool_input", {}) or {}).get("command", "") or ""
if not any(b in cmd for b in BENCH):
    raise SystemExit(0)
changed = (git("diff", "--name-only", "HEAD") + git("ls-files", "--others", "--exclude-standard")).split()
touched = sorted({axis for axis, prefixes in AXES.items()
                  for f in changed if any(f.startswith(p) for p in prefixes)})
if len(touched) > 1:
    deny(f"Single-axis discipline: uncommitted changes span axes {touched}. Commit or stash all but one "
         "before benchmarking, and keep model/temperature identical to the baseline.")
raise SystemExit(0)
