#!/usr/bin/env python3
"""H7 — PreToolUse Bash: replay-mode tests run with network isolation (S2-T00b exit criterion).

Linux: prefix `unshare -n` (needs user namespaces or root). Otherwise: dead-proxy env vars, which only
catch proxy-honouring clients (httpx, requests, yfinance, edgartools) — raw sockets are NOT caught;
the real assertion belongs in the test itself.
"""
from __future__ import annotations

import platform
import shutil

from _common import pre_tool_decision, read_event


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"

ev = read_event()
ti = ev.get("tool_input", {}) or {}
cmd = ti.get("command", "") or ""
is_replay = "pytest" in cmd and ("-m replay" in cmd or "EVIDENCE_MODE=replay" in cmd or "tests/replay" in cmd)
if not is_replay or "unshare -n" in cmd or "NO_NETWORK_APPLIED=1" in cmd:
    raise SystemExit(0)
if platform.system() == "Linux" and shutil.which("unshare"):
    new_cmd = f"NO_NETWORK_APPLIED=1 unshare -n sh -c {_q(cmd)}"
    note = "network namespace disabled via unshare -n"
else:
    new_cmd = ("NO_NETWORK_APPLIED=1 HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 "
               f"NO_PROXY= {cmd}")
    note = "dead-proxy fallback (raw sockets not blocked)"
pre_tool_decision("allow", f"replay test isolated: {note}", updated_input={**ti, "command": new_cmd},
                  context=f"H7 rewrote the command for zero-network replay ({note}). Record the rewritten command in the task Result.")
