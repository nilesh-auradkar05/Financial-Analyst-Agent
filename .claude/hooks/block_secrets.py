#!/usr/bin/env python3
"""H8 — PreToolUse Bash|Read: block reads of secret files and credential patterns in commands."""
from __future__ import annotations

import re

from _common import deny, read_event, rel

SECRET_PATHS = re.compile(r"(^|/)(\.env(\.[\w-]+)?|\.aws/credentials|\.aws/config|id_rsa|id_ed25519|.*\.pem|.*\.p12|"
                          r"\.netrc|secrets?\.(json|ya?ml|toml))$")
CRED_PATTERNS = re.compile(r"(AKIA[0-9A-Z]{16}|tvly-[A-Za-z0-9]{20,}|lsv2_[a-z]{2}_[0-9a-f]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|"
                           r"AWS_SECRET_ACCESS_KEY\s*=\s*\S{20,})")
READ_CMDS = re.compile(r"\b(cat|less|more|head|tail|grep|rg|sed|awk|bat|xxd|strings|source|\.)\b")

ev = read_event()
tool = ev.get("tool_name", "")
ti = ev.get("tool_input", {}) or {}
if tool == "Read":
    if SECRET_PATHS.search(rel(ti.get("file_path", ""))):
        deny("Secrets file. Use .env.example / documented config names instead.")
    raise SystemExit(0)
cmd = ti.get("command", "") or ""
if CRED_PATTERNS.search(cmd):
    deny("Command contains a credential-shaped literal. Never place secrets in commands or files; use env vars.")
if READ_CMDS.search(cmd):
    for tok in re.findall(r"[\w./~-]+", cmd):
        if SECRET_PATHS.search(tok):
            deny(f"Reading secrets file '{tok}' is blocked.")
raise SystemExit(0)
