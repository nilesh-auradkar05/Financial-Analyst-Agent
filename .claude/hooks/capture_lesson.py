#!/usr/bin/env python3
"""H11 — PostToolUse Bash: when a failure follows a user correction, append a lesson stub.

Heuristic: the transcript's last human turn contains a correction phrase and this Bash result
contains a failure marker. Appends a dated stub to tasks/lessons.md for the agent to complete.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from _common import REPO, feedback, read_event

CORRECTION = re.compile(r"\b(no[,.]|wrong|that's not|that is not|incorrect|you misdiagnosed|not what i|stop)\b", re.I)
FAILURE = re.compile(r"(FAILED|Traceback|Error:|error\[|exit code [1-9])")

ev = read_event()
resp = ev.get("tool_response", {}) or {}
output = resp if isinstance(resp, str) else json.dumps(resp)
if not FAILURE.search(output):
    raise SystemExit(0)
tp = ev.get("transcript_path") or ""
transcript = Path(tp)
last_human = ""
if tp and transcript.is_file():
    for line in reversed(transcript.read_text(errors="ignore").splitlines()[-400:]):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "user" or rec.get("role") == "user":
            msg = rec.get("message", rec)
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            last_human = content if isinstance(content, str) else " ".join(
                c.get("text", "") for c in content if isinstance(c, dict))
            break
if not CORRECTION.search(last_human):
    raise SystemExit(0)
lessons = REPO / "tasks" / "lessons.md"
lessons.parent.mkdir(exist_ok=True)
stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
stub = (f"\n## {stamp} — (fill in) pattern behind the correction\n"
        f"- Correction: {last_human.strip()[:200]}\n- Failure seen: {FAILURE.search(output).group(0)}\n"
        "- Rule for next time: \n")
with lessons.open("a") as f:
    f.write(stub)
feedback("H11: a user correction was followed by a failure. A stub was appended to tasks/lessons.md — "
         "complete the 'Rule for next time' line before continuing.")
