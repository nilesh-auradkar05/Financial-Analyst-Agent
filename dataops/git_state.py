from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def git_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]

    def run(args: list[str]) -> str:
        try:
            return subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repo_root}",
                    "-C",
                    str(repo_root),
                    *args,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run(["rev-parse", "--short", "HEAD"]) or "unknown",
        "dirty": bool(run(["status", "--porcelain"])),
    }


def current_code_version() -> str:
    return str(git_state()["commit"])
