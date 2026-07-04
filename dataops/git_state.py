from __future__ import annotations

import subprocess
from typing import Any


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def current_code_version() -> str:
    return str(git_state()["commit"])
