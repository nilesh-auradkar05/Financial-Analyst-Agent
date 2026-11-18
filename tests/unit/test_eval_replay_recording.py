"""quality_baseline records its drafts and can approve a baseline (docs/test-plan.md §14, S6.2).

A replay run of ``python -m evaluation.quality_baseline`` (frozen evidence release built through the
real dataops APIs, memo model faked at the provider boundary) runs in a throwaway working
directory. The run's result file must carry each scored draft so the gate can re-score it, and
``--classification approved`` must be accepted only for a token-checked replay run.
"""

from __future__ import annotations

import json
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from app.agents import graph
from app.verification.grounding import evaluate_memo_grounding
from evaluation import quality_baseline
from tests.replay.support import BuiltRelease, FakeChat, build_release


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any non-loopback, non-AF_UNIX connect raises: these runs are offline."""
    real_connect = socket.socket.connect

    def refuse(address: Any) -> NoReturn:
        raise OSError(f"network access attempted in an offline test: {address!r}")

    def guarded(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
            return
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "localhost", "::1"):
            real_connect(self, address)
            return
        refuse(address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "work"
    directory.mkdir()
    monkeypatch.chdir(directory)
    return directory


@pytest.fixture
def release(tmp_path: Path) -> BuiltRelease:
    return build_release(tmp_path / "release")


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> FakeChat:
    model = FakeChat()
    monkeypatch.setattr(graph, "get_llm", lambda _settings: model)
    return model


def argv_for(release: BuiltRelease, *extra: str) -> list[str]:
    return [
        "quality_baseline",
        "--tickers",
        "AAPL",
        "MSFT",
        "--repeats",
        "2",
        "--evidence-release",
        release.release_id,
        "--registry-root",
        str(release.registry_root),
        *extra,
    ]


def written_run(workdir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    [result_file] = sorted((workdir / "evaluation" / "quality_res").glob("*.json"))
    [record_file] = sorted((workdir / "evaluation" / "registry" / "runs").glob("*.json"))
    return json.loads(result_file.read_text()), json.loads(record_file.read_text())


def run_gate(argv: list[str]) -> Any:
    from evaluation import eval_replay

    try:
        return eval_replay.main(argv)
    except SystemExit as exit_:
        return exit_.code


async def test_recorded_drafts_reproduce_each_runs_rates_and_the_approved_baseline_passes_the_gate(
    monkeypatch: pytest.MonkeyPatch, workdir: Path, release: BuiltRelease, chat: FakeChat
) -> None:
    """§14 Recorded drafts (S6.2): re-scoring every recorded run reproduces its rates; approved.

    The gate is then run against the record the run wrote, end to end.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        argv_for(release, "--checker", "token", "--classification", "approved"),
    )

    await quality_baseline.main()

    result, record = written_run(workdir)
    assert record["classification"] == "approved"
    assert result["grounding_checker"] == "token"
    assert len(result["raw_runs"]) == 4
    for run in result["raw_runs"]:
        assert isinstance(run["recorded_memo"], str) and run["recorded_memo"].strip()
        assert isinstance(run["recorded_registry"], list) and run["recorded_registry"]
        verdict = evaluate_memo_grounding(run["recorded_memo"], run["recorded_registry"])
        assert verdict.grounded_claim_rate == run["grounded_claim_rate"]
        assert verdict.citation_coverage_rate == run["citation_coverage_rate"]

    gate = run_gate(
        [
            "--baseline",
            record["run_id"],
            "--registry-root",
            str(workdir / "evaluation" / "registry" / "runs"),
        ]
    )
    assert gate == 0


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (["--checker", "token", "--classification", "approved"], "evidence"),
        (["--classification", "approved", "--checker", "semantic"], "token"),
        (["--classification", "approved"], "token"),
    ],
    ids=[
        "approved-without-evidence-release",
        "approved-with-semantic-checker",
        "approved-with-default-checker",
    ],
)
async def test_approved_is_refused_before_any_model_call_and_nothing_is_written(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    release: BuiltRelease,
    chat: FakeChat,
    extra: list[str],
    reason: str,
) -> None:
    """§14 Recorded drafts (S6.2): --classification approved needs --evidence-release and token."""
    argv = argv_for(release, *extra)
    if reason == "evidence":
        at = argv.index("--evidence-release")
        del argv[at : at + 2]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises((Exception, SystemExit)) as refused:
        await quality_baseline.main()

    captured = capsys.readouterr()
    text = f"{refused.value} {captured.out} {captured.err}".lower()
    assert "unrecognized arguments" not in text, "the flag must exist and refuse on its merits"
    assert reason in text
    assert chat.prompts_seen == []
    assert not (workdir / "evaluation").exists()
