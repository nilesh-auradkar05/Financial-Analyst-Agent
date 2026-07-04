from __future__ import annotations

import subprocess

from dataops.git_state import git_state


def test_git_state_survives_temp_home_without_safe_directory_config(monkeypatch, tmp_path) -> None:
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    monkeypatch.setenv("HOME", str(tmp_path))

    assert git_state()["commit"] == expected
