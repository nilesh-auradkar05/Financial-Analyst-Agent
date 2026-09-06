#!/usr/bin/env python3
"""H3/H4/H5 — PreToolUse Edit|Write: protect governance docs, frozen fixtures, and eval-result lineage."""
from __future__ import annotations

import os
import re

from _common import deny, read_event, rel

GOVERNANCE = re.compile(r"^docs/(SPEC\.md|SPEC-AMENDMENT.*\.md|test-plan\.md|retrieval-benchmark\.md|"
                        r"LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN(_v\d+)?\.md|adr/.*)$")
FROZEN = re.compile(r"^evaluation/(fixtures/[^/]*_v\d+\.json|dataset/.*|evidence_releases/.*/manifest\.json)$")
EVAL_RESULT = re.compile(r"^evaluation/([\w-]+_res|registry/runs)/.*\.json$")
LINEAGE = ("git_sha", "model", "temperature", "prompt_version", "evidence_release", "snapshot_hash", "fixture_version")

ev = read_event()
ti = ev.get("tool_input", {}) or {}
path = rel(ti.get("file_path") or ti.get("path") or "")
if not path:
    raise SystemExit(0)

if GOVERNANCE.match(path) and os.environ.get("ALLOW_SPEC_EDIT") != "1":
    deny(f"{path} is a governance document (SPEC §0 rank ≤ 3). Amend via ADR/amendment with ALLOW_SPEC_EDIT=1; "
         "code follows docs, not the reverse.")

if FROZEN.match(path):
    deny(f"{path} is a frozen fixture/dataset. Create a new *_vN+1 (or *_candidate) file; never edit the oracle to fit code.")

if EVAL_RESULT.match(path):
    body = ti.get("content") or ti.get("new_string") or ""
    if ti.get("edits"):  # MultiEdit
        body = " ".join(e.get("new_string", "") for e in ti["edits"])
    missing = [k for k in LINEAGE if f'"{k}"' not in body]
    if missing:
        deny(f"{path}: eval result lacks lineage keys {missing} (SPEC §11.2). Emit via the registry writer.")
raise SystemExit(0)
