"""Eval producers write lineage (docs/test-plan.md §14 "Producers write lineage", S6.1).

``python -m evaluation.quality_baseline`` (replay of a frozen evidence release, memo model faked at
the provider boundary) and ``python -m evaluation.run_rag_quality_eval`` (no judges) run in a
throwaway working directory. Their relative output paths land there, so the test reads back the
result under ``evaluation/<name>_res/`` and the record under ``evaluation/registry/runs/``.
``evaluation.registry`` is imported inside the tests, after the files are checked, so a red run
fails on the missing output rather than on the import.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from app.agents import graph
from evaluation import quality_baseline, run_rag_quality_eval
from tests.replay.support import FakeChat, build_release
from tests.unit.test_judge_ladder import Ladder, ladder  # noqa: F401  (pytest fixture)

LINEAGE_KEYS = (
    "run_id",
    "git_sha",
    "branch",
    "model",
    "temperature",
    "prompt_version",
    "evidence_release",
    "snapshot_hash",
    "fixture_version",
)
HEX16 = re.compile(r"^[0-9a-f]{16}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any non-AF_UNIX connect raises: these runs are offline."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
            return
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "localhost", "::1"):
            real_connect(self, address)
            return
        refuse(address)

    def refuse(address: Any) -> NoReturn:
        raise OSError(f"network access attempted in an offline test: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield


def results_in(workdir: Path, name: str) -> list[Path]:
    directory = workdir / "evaluation" / name
    return sorted(directory.glob("*.json")) if directory.exists() else []


def registry_records(workdir: Path) -> list[Path]:
    directory = workdir / "evaluation" / "registry" / "runs"
    return sorted(directory.glob("*.json")) if directory.exists() else []


def assert_one_lineaged_result_and_record(
    workdir: Path, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exactly one result and one candidate record that name each other; returns both."""
    [result_file] = results_in(workdir, name)
    [record_file] = registry_records(workdir)
    result = json.loads(result_file.read_text())
    record = json.loads(record_file.read_text())

    for key in LINEAGE_KEYS:
        assert key in result, f"result lacks lineage key {key}"
        assert result[key] not in ("", None), f"result lineage key {key} is empty"
        assert record[key] == result[key], f"result and registry record disagree on {key}"
    assert record_file.name == f"{record['run_id']}.json"
    assert (workdir / record["result_path"]).resolve() == result_file.resolve()
    assert record["classification"] == "candidate"
    assert isinstance(record["metrics"], dict)
    assert isinstance(result["temperature"], float | int)

    from evaluation.registry import prompt_version

    assert result["prompt_version"] == prompt_version()
    assert HEX16.fullmatch(result["fixture_version"])
    assert HEX64.fullmatch(result["snapshot_hash"])
    return result, record


# --- quality_baseline ------------------------------------------------------------------------


async def test_quality_baseline_replay_writes_a_lineaged_result_and_one_registry_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§14 Producers write lineage: a quality_baseline replay run."""
    built = build_release(tmp_path / "release")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(graph, "get_llm", lambda _settings: FakeChat())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quality_baseline",
            "--tickers",
            "AAPL",
            "--repeats",
            "1",
            "--checker",
            "token",
            "--evidence-release",
            built.release_id,
            "--registry-root",
            str(built.registry_root),
        ],
    )

    await quality_baseline.main()

    result, record = assert_one_lineaged_result_and_record(workdir, "quality_res")
    from evaluation.registry import run_snapshot_hash

    assert result["evidence_release"] == built.release_id
    assert result["snapshot_hash"] == run_snapshot_hash(
        built.release_id, built.manifest.snapshot_ids
    )
    assert record["metrics"]
    assert all(isinstance(value, float | int) for value in record["metrics"].values())
    assert {"grounded_claim_rate", "citation_coverage"} <= record["metrics"].keys()


# --- run_rag_quality_eval --------------------------------------------------------------------


def rag_sample(case_id: str, snapshot_hash: str | None) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "case_id": case_id,
        "ticker": "AAPL",
        "input": "What drove revenue?",
        "actual_output": "Services revenue grew.",
        "expected_output": "Services revenue grew.",
        "retrieval_context": ["Services revenue grew 14%."],
        "context": ["Services revenue grew 14%."],
        "metadata": {} if snapshot_hash is None else {"snapshot_hash": snapshot_hash},
    }
    if snapshot_hash == "":
        sample["metadata"] = {"snapshot_hash": ""}
    return sample


def write_samples(path: Path, samples: list[dict[str, Any]]) -> bytes:
    path.write_text(json.dumps(samples))
    return path.read_bytes()


def test_rag_eval_writes_a_lineaged_result_and_one_registry_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§14 Producers write lineage: a run_rag_quality_eval run with no judges, no network."""
    monkeypatch.chdir(tmp_path)
    case_hashes = [hashlib.sha256(name.encode()).hexdigest() for name in ("case-1", "case-2")]
    raw = write_samples(
        tmp_path / "samples.json",
        [rag_sample("case-1", case_hashes[0]), rag_sample("case-2", case_hashes[1])],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rag_quality_eval",
            "samples.json",
            "--judges",
            "",
            "--judge-model",
            "fake-judge",
            "--judge-temperature",
            "0.25",
        ],
    )

    run_rag_quality_eval.main()

    result, record = assert_one_lineaged_result_and_record(tmp_path, "rag_quality_res")
    from evaluation.registry import run_snapshot_hash

    assert result["evidence_release"] == "live"
    assert result["snapshot_hash"] == run_snapshot_hash("live", case_hashes)
    assert result["model"] == "fake-judge"
    assert result["temperature"] == 0.25
    assert result["fixture_version"] == hashlib.sha256(raw).hexdigest()[:16]
    assert record["snapshot_hash"] == result["snapshot_hash"]


@pytest.mark.parametrize(
    "bad_sample",
    [rag_sample("case-2", None), rag_sample("case-2", "")],
    ids=["metadata-without-snapshot-hash", "empty-snapshot-hash"],
)
def test_rag_eval_refuses_samples_without_a_snapshot_hash_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad_sample: dict[str, Any],
) -> None:
    """§14 Producers write lineage: one sample lacking metadata.snapshot_hash refuses the run."""
    monkeypatch.chdir(tmp_path)
    write_samples(tmp_path / "samples.json", [rag_sample("case-1", "a" * 64), bad_sample])
    monkeypatch.setattr(sys, "argv", ["run_rag_quality_eval", "samples.json", "--judges", ""])

    with pytest.raises((Exception, SystemExit)) as refused:
        run_rag_quality_eval.main()

    captured = capsys.readouterr()
    assert "snapshot_hash" in f"{refused.value} {captured.out} {captured.err}"
    assert not (tmp_path / "evaluation").exists()


def test_rag_eval_refuses_before_any_judge_runs(
    ladder: Ladder,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§14 Producers write lineage: the refusal comes before any judge call (loopback judge)."""
    ladder.configure()
    write_samples(
        tmp_path / "samples.json", [rag_sample("case-1", "a" * 64), rag_sample("case-2", None)]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rag_quality_eval",
            "samples.json",
            "--judges",
            "deepeval",
            "--judge-provider",
            "opus",
        ],
    )

    with pytest.raises((Exception, SystemExit)):
        run_rag_quality_eval.main()

    assert ladder.foundry.requests == []
    assert ladder.bedrock.requests == []
    assert not (tmp_path / "evaluation").exists()
