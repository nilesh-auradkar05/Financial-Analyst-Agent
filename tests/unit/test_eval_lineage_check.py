"""Hook H5 and the CI lineage check agree (docs/test-plan.md §14 "Hook and CI agree", S6.1).

Both are observed as processes: H5 gets a Write event on stdin and answers allow or deny;
``scripts/ci/check_eval_lineage.py`` scans a root directory and exits 0 or 1 naming offenders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_hooks import decision, run_hook, write

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "check_eval_lineage.py"
LINEAGE = (
    "git_sha",
    "model",
    "temperature",
    "prompt_version",
    "evidence_release",
    "snapshot_hash",
    "fixture_version",
)


def body(*, without: str | None = None, extra: dict[str, object] | None = None) -> str:
    payload: dict[str, object] = {key: "v" for key in LINEAGE if key != without}
    payload.update(extra or {})
    return json.dumps(payload)


def run_check(root: Path | None, cwd: Path) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(SCRIPT)]
    if root is not None:
        args += ["--root", str(root)]
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=120)


def check_verdict(root: Path | None, cwd: Path) -> tuple[str, subprocess.CompletedProcess[str]]:
    proc = run_check(root, cwd)
    assert proc.returncode in (0, 1), proc.stderr
    return ("deny" if proc.returncode == 1 else "allow"), proc


def hook_verdict(project: Path, path: str, content: str) -> str:
    proc = run_hook("protect_paths.py", write(path, content), project)
    if not proc.stdout.strip():
        assert proc.returncode == 0, proc.stderr
        return "allow"
    return str(decision(proc)["permissionDecision"])


SAMPLES: list[tuple[str, str, str]] = [
    ("evaluation/quality_res/x.json", body(), "allow"),
    ("evaluation/rag_quality_res/x.json", body(), "allow"),
    ("evaluation/registry/runs/r.json", body(), "allow"),
    ("evaluation/quality_res/x.json", body(extra={"run_id": "r"}), "allow"),
    ("evaluation/my-run_res/nested/x.json", body(), "allow"),
    *[
        (path, body(without=key), "deny")
        for key in LINEAGE
        for path in (
            "evaluation/quality_res/x.json",
            "evaluation/registry/runs/r.json",
        )
    ],
    ("evaluation/quality_res/x.json", "{}", "deny"),
    ("evaluation/quality_res/sub/x.json", body(without="model"), "deny"),
    ("evaluation/latency_res/x.json", body(without="git_sha"), "deny"),
    ("evaluation/results/x.json", "{}", "allow"),
    ("evaluation/archive/quality_res/x.json", "{}", "allow"),
    ("evaluation/quality_res/notes.txt", "no keys", "allow"),
    ("evaluation/registry/other/r.json", "{}", "allow"),
]


@pytest.mark.parametrize(
    ("path", "content", "expected"),
    SAMPLES,
    ids=[f"{expected}:{path}:{index}" for index, (path, _, expected) in enumerate(SAMPLES)],
)
def test_hook_and_ci_check_give_the_same_decision(
    tmp_path: Path, path: str, content: str, expected: str
) -> None:
    """§14 Hook and CI agree: the same (path, body) is allowed or denied by both."""
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(content)

    ci_verdict, proc = check_verdict(tmp_path, tmp_path)
    hook = hook_verdict(tmp_path, path, content)

    assert hook == expected
    assert ci_verdict == expected
    if expected == "deny":
        assert path in proc.stdout
    else:
        assert proc.stdout.strip() == "" or path not in proc.stdout


def test_ci_check_passes_on_a_root_without_evaluation_results(tmp_path: Path) -> None:
    """§14 Hook and CI agree: nothing under the patterns means nothing to refuse."""
    verdict, proc = check_verdict(tmp_path, tmp_path)

    assert verdict == "allow", proc.stdout


def test_ci_check_names_only_the_planted_unlineaged_result(tmp_path: Path) -> None:
    """§14 Hook and CI agree: fails, naming the file, on a planted *_res result lacking a key."""
    good = tmp_path / "evaluation" / "quality_res" / "good.json"
    bad = tmp_path / "evaluation" / "rag_quality_res" / "bad.json"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text(body())
    bad.write_text(body(without="snapshot_hash"))

    verdict, proc = check_verdict(tmp_path, tmp_path)

    assert verdict == "deny"
    assert "evaluation/rag_quality_res/bad.json" in proc.stdout
    assert "good.json" not in proc.stdout


def test_ci_check_passes_on_the_real_repository() -> None:
    """§14 Hook and CI agree: the repository itself is clean (legacy results live in
    evaluation/archive/, outside both patterns). Run with the default root."""
    verdict, proc = check_verdict(None, REPO)

    assert verdict == "allow", proc.stdout
