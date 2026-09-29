"""Claude Code agent-tooling hooks (docs/test-plan.md §15).

Each hook under ``.claude/hooks/`` is exercised by piping a sample event JSON to the script as a
subprocess and asserting stdout / exit code. Every test runs against a throwaway project directory
(``CLAUDE_PROJECT_DIR``), never the real repo state.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def run_hook(
    name: str,
    event: dict[str, Any],
    project: Path,
    extra_env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ALLOW_SPEC_EDIT", "SKIP_STOP_GATE")}
    env["CLAUDE_PROJECT_DIR"] = str(project)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        timeout=timeout,
    )


def decision(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the PreToolUse ``hookSpecificOutput`` object from stdout."""
    assert proc.returncode == 0, proc.stderr
    out: dict[str, Any] = json.loads(proc.stdout)["hookSpecificOutput"]
    return out


def assert_denied(proc: subprocess.CompletedProcess[str]) -> str:
    out = decision(proc)
    assert out["permissionDecision"] == "deny"
    return str(out["permissionDecisionReason"])


def assert_silent_allow(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def bash(cmd: str) -> dict[str, Any]:
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def write(path: str, content: str = "x") -> dict[str, Any]:
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=project, check=True, capture_output=True, env={**os.environ, **GIT_ENV}
    )


def make_repo(project: Path, files: dict[str, str] | None = None) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    git(project, "init", "-q")
    for rel_path, body in {"README.md": "hi\n", **(files or {})}.items():
        target = project / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "init")
    return project


# --- H2 require_plan -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("todo", "prompt", "blocked"),
    [
        ("# Todo\n- [x] done item\n", "implement the retriever", True),  # H2 deny: no open item
        (None, "implement the retriever", True),  # H2 deny: no todo file at all
        ("# Todo\n- [x] done item\n", "explain the retriever", False),  # H2 allow: explanatory
        ("# Todo\n- [ ] open item\n", "implement the retriever", False),  # H2 allow: open plan item
    ],
)
def test_h2_require_plan(tmp_path: Path, todo: str | None, prompt: str, blocked: bool) -> None:
    if todo is not None:
        (tmp_path / "tasks").mkdir()
        (tmp_path / "tasks" / "todo.md").write_text(todo)
    proc = run_hook("require_plan.py", {"prompt": prompt}, tmp_path)
    assert proc.returncode == 0
    if blocked:
        out = json.loads(proc.stdout)
        assert out["decision"] == "block"
        assert "tasks/todo.md" in out["reason"]
    else:
        assert proc.stdout.strip() == ""


# --- H3 / H4 / H5 protect_paths --------------------------------------------------------------

ALL_LINEAGE = (
    '{"git_sha": "a", "model": "m", "temperature": 0, "prompt_version": "p", '
    '"evidence_release": "e", "snapshot_hash": "h", "fixture_version": "v"}'
)


def test_h3_governance_doc_denied_without_env(tmp_path: Path) -> None:
    reason = assert_denied(run_hook("protect_paths.py", write("docs/SPEC.md"), tmp_path))
    assert "docs/SPEC.md" in reason and "ALLOW_SPEC_EDIT" in reason


def test_h3_governance_doc_allowed_with_env(tmp_path: Path) -> None:
    proc = run_hook("protect_paths.py", write("docs/SPEC.md"), tmp_path, {"ALLOW_SPEC_EDIT": "1"})
    assert_silent_allow(proc)


def test_h4_frozen_fixture_denied(tmp_path: Path) -> None:
    ev = write("evaluation/fixtures/retrieval_shared_benchmark_v1.json")
    assert "frozen" in assert_denied(run_hook("protect_paths.py", ev, tmp_path))


def test_h4_candidate_fixture_allowed(tmp_path: Path) -> None:
    ev = write("evaluation/fixtures/retrieval_shared_benchmark_v3_candidate.json")
    assert_silent_allow(run_hook("protect_paths.py", ev, tmp_path))


def test_h5_eval_result_missing_lineage_denied(tmp_path: Path) -> None:
    ev = write("evaluation/quality_res/x.json", '{"score": 1}')
    reason = assert_denied(run_hook("protect_paths.py", ev, tmp_path))
    assert "snapshot_hash" in reason


def test_h5_eval_result_partial_lineage_names_missing_key(tmp_path: Path) -> None:
    body = ALL_LINEAGE.replace('"snapshot_hash": "h", ', "")
    reason = assert_denied(
        run_hook("protect_paths.py", write("evaluation/quality_res/x.json", body), tmp_path)
    )
    assert "snapshot_hash" in reason and "git_sha" not in reason


def test_h5_eval_result_full_lineage_allowed(tmp_path: Path) -> None:
    ev = write("evaluation/quality_res/x.json", ALL_LINEAGE)
    assert_silent_allow(run_hook("protect_paths.py", ev, tmp_path))


# --- H6 single_axis --------------------------------------------------------------------------

BENCH_CMD = "python evaluation/run_rag_quality_eval.py --fixture f.json"


@needs_git
@pytest.mark.parametrize(
    ("dirty", "cmd", "denied"),
    [
        (["app/prompts/a.txt", "app/agents/g.py"], BENCH_CMD, True),  # H6 deny: prompt + graph
        (["app/prompts/a.txt"], BENCH_CMD, False),  # H6 allow: one axis
        (["app/prompts/a.txt", "app/agents/g.py"], "ls -la", False),  # not a benchmark command
    ],
)
def test_h6_single_axis(tmp_path: Path, dirty: list[str], cmd: str, denied: bool) -> None:
    make_repo(tmp_path)
    for rel_path in dirty:  # untracked files count as uncommitted changes
        (tmp_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel_path).write_text("x")
    proc = run_hook("single_axis.py", bash(cmd), tmp_path)
    if denied:
        reason = assert_denied(proc)
        assert "prompt" in reason and "graph" in reason
    else:
        assert_silent_allow(proc)


# --- H7 replay_no_network --------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["pytest -m replay tests/", "EVIDENCE_MODE=replay pytest tests/x.py"]
)
def test_h7_replay_command_rewritten_with_isolation(tmp_path: Path, cmd: str) -> None:
    """Only the rewritten command text is checked; the rewritten command is never executed."""
    out = decision(run_hook("replay_no_network.py", bash(cmd), tmp_path))
    assert out["permissionDecision"] == "allow"
    new_cmd = out["updatedInput"]["command"]
    assert "NO_NETWORK_APPLIED=1" in new_cmd
    assert "unshare -n" in new_cmd or "HTTP_PROXY=http://127.0.0.1:9" in new_cmd
    assert "pytest" in new_cmd and "replay" in new_cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "uv run pytest tests/unit -q",
        "NO_NETWORK_APPLIED=1 pytest -m replay",
        "unshare -n pytest -m replay",
    ],
)
def test_h7_non_replay_or_already_isolated_untouched(tmp_path: Path, cmd: str) -> None:
    assert_silent_allow(run_hook("replay_no_network.py", bash(cmd), tmp_path))


# --- H8 block_secrets ------------------------------------------------------------------------


def fake_aws_key() -> str:
    return "AK" + "IA" + "X" * 16  # built at runtime so this file holds no literal key


@pytest.mark.parametrize(
    "event",
    [
        bash("cat .env"),  # H8 deny: secrets file read
        bash(f"echo {fake_aws_key()} > notes.txt"),  # H8 deny: credential literal
        {"tool_name": "Read", "tool_input": {"file_path": ".env"}},  # H8 deny: Read tool
    ],
)
def test_h8_secrets_denied(tmp_path: Path, event: dict[str, Any]) -> None:
    assert_denied(run_hook("block_secrets.py", event, tmp_path))


@pytest.mark.parametrize(
    "event",
    [bash("cat README.md"), {"tool_name": "Read", "tool_input": {"file_path": "README.md"}}],
)
def test_h8_ordinary_read_allowed(tmp_path: Path, event: dict[str, Any]) -> None:
    assert_silent_allow(run_hook("block_secrets.py", event, tmp_path))


# --- H9 block_destructive --------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["git push --force origin main", "git push origin main -f", "rm -rf /home"]
)
def test_h9_destructive_denied(tmp_path: Path, cmd: str) -> None:
    assert "Blocked" in assert_denied(run_hook("block_destructive.py", bash(cmd), tmp_path))


@pytest.mark.parametrize("cmd", ["rm -rf /tmp/x", "git push origin main", "git status"])
def test_h9_safe_command_allowed(tmp_path: Path, cmd: str) -> None:
    assert_silent_allow(run_hook("block_destructive.py", bash(cmd), tmp_path))


# --- H12 stop_gate ---------------------------------------------------------------------------


def fake_uv_dir(tmp_path: Path) -> Path:
    """A `uv` shim (outside the project) that runs `uv run <tool> ...` with this interpreter's env,
    so the gate's `uv run pytest` runs the throwaway project's tests instead of the real suite."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "uv"
    shim.write_text(f'#!/bin/sh\nshift\nshift\nexec "{sys.executable}" -m pytest "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return bindir


def stop_project(tmp_path: Path, test_body: str) -> Path:
    """Two commits so the last commit touches tests/, which makes the gate run tests/unit."""
    project = make_repo(tmp_path / "proj")
    (project / "tests" / "unit").mkdir(parents=True)
    (project / "tests" / "unit" / "test_x.py").write_text(test_body)
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "add tests")
    return project


def stop_env(tmp_path: Path) -> dict[str, str]:
    return {"PATH": f"{fake_uv_dir(tmp_path)}{os.pathsep}{os.environ['PATH']}"}


@needs_git
def test_h12_dirty_tree_blocks(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "proj")
    (project / "stray.txt").write_text("x")
    proc = run_hook("stop_gate.py", {"hook_event_name": "Stop"}, project)
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["decision"] == "block"
    assert "uncommitted changes" in out["reason"] and "stray.txt" in out["reason"]


@needs_git
def test_h12_dirty_reason_keeps_full_path_of_first_modified_file(tmp_path: Path) -> None:
    """H12 regression: an unstaged edit listed first (" M .cfg") must be reported as ".cfg", not "cfg"."""
    project = make_repo(tmp_path / "proj", {".cfg": "a\n"})
    (project / ".cfg").write_text("b\n")
    out = json.loads(run_hook("stop_gate.py", {"hook_event_name": "Stop"}, project).stdout)
    assert out["decision"] == "block"
    assert "\n    .cfg" in out["reason"]


@needs_git
def test_h12_stop_hook_active_exits_immediately(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "proj")
    (project / "stray.txt").write_text("x")  # would block if the loop guard did not fire
    assert_silent_allow(run_hook("stop_gate.py", {"stop_hook_active": True}, project))


@needs_git
def test_h12_clean_tree_and_green_tests_allow(tmp_path: Path) -> None:
    project = stop_project(tmp_path, "def test_ok():\n    assert True\n")
    assert_silent_allow(run_hook("stop_gate.py", {}, project, stop_env(tmp_path)))


@needs_git
def test_h12_failing_unit_tests_block(tmp_path: Path) -> None:
    project = stop_project(tmp_path, "def test_bad():\n    assert False\n")
    proc = run_hook("stop_gate.py", {}, project, stop_env(tmp_path))
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["decision"] == "block"
    assert "unit tests failing" in out["reason"]
    assert "uncommitted changes" not in out["reason"]
