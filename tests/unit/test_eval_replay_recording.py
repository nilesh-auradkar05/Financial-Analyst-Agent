"""quality_baseline records its drafts and can approve a baseline (docs/test-plan.md §14, S6.2).

A replay run of ``python -m evaluation.quality_baseline`` (frozen evidence release built through the
real dataops APIs, memo model faked at the provider boundary) runs in a throwaway working
directory. The run's result file must carry each scored draft so the gate can re-score it, and
``--classification approved`` must be accepted only for a token-checked replay run.
"""

from __future__ import annotations

import copy
import json
import re
import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from app.agents import graph
from app.verification.grounding import evaluate_memo_grounding
from evaluation import quality_baseline
from tests.replay.support import (
    CLEAN,
    FIRST_DRAFT_FAILS,
    BuiltRelease,
    FakeChat,
    ScriptedChat,
    Turn,
    build_release,
)


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
    """Exit code of the gate: its return value, or the code of the SystemExit it raises."""
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


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[Turn]], ScriptedChat]:
    """Install a provider that follows a plan; the real verifier decides pass or fail."""

    def install(plan: list[Turn]) -> ScriptedChat:
        model = ScriptedChat(plan)
        monkeypatch.setattr(graph, "get_llm", lambda _settings: model)
        return model

    return install


def single_ticker_argv(release: BuiltRelease) -> list[str]:
    argv = argv_for(release, "--checker", "token")
    at = argv.index("--tickers")
    return [*argv[: at + 1], "AAPL", *argv[at + 3 :]]


async def run_one_ticker_twice(
    monkeypatch: pytest.MonkeyPatch, release: BuiltRelease, workdir: Path
) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", single_ticker_argv(release))
    await quality_baseline.main()
    result, _record = written_run(workdir)
    return result


async def test_every_raw_run_carries_its_repeat_and_its_drafts(
    monkeypatch: pytest.MonkeyPatch, workdir: Path, release: BuiltRelease, chat: FakeChat
) -> None:
    """§14 Recorded drafts (S6.4): each raw run carries `repeat` (0-based within its ticker),
    `draft_count` and `draft_history`, one history entry per draft."""
    monkeypatch.setattr(sys, "argv", argv_for(release, "--checker", "token"))

    await quality_baseline.main()

    result, _record = written_run(workdir)
    assert sorted((run["ticker"], run["repeat"]) for run in result["raw_runs"]) == [
        ("AAPL", 0),
        ("AAPL", 1),
        ("MSFT", 0),
        ("MSFT", 1),
    ]
    for run in result["raw_runs"]:
        assert run["draft_count"] >= 1
        assert len(run["draft_history"]) == run["draft_count"]
        for entry in run["draft_history"]:
            assert isinstance(entry["grounded_claim_rate"], float)
            assert isinstance(entry["passed"], bool)


async def test_a_run_whose_first_draft_fails_records_two_drafts_and_the_passing_memo(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    release: BuiltRelease,
    scripted: Callable[[list[Turn]], ScriptedChat],
) -> None:
    """§14 Recorded drafts (S6.4): the replay workflow's revise loop runs inside the baseline; a
    run that fails draft 1 and passes the revision records 2 drafts and the passing memo, and
    re-scoring that memo reproduces the run's rate."""
    model = scripted([FIRST_DRAFT_FAILS, CLEAN, CLEAN])

    result = await run_one_ticker_twice(monkeypatch, release, workdir)

    runs = result["raw_runs"]
    assert len(model.prompts_seen) == 3
    assert sorted(run["repeat"] for run in runs) == [0, 1]
    [revised] = [run for run in runs if run["draft_count"] == 2]
    [plain] = [run for run in runs if run["draft_count"] == 1]

    assert [entry["passed"] for entry in revised["draft_history"]] == [False, True]
    first, second = (entry["grounded_claim_rate"] for entry in revised["draft_history"])
    assert first < second == revised["grounded_claim_rate"]
    assert revised["recorded_memo"] in model.memos
    assert not re.search(r"\b81\d\d%", revised["recorded_memo"]), "the failed draft was recorded"
    verdict = evaluate_memo_grounding(revised["recorded_memo"], revised["recorded_registry"])
    assert verdict.passed is True
    assert verdict.grounded_claim_rate == revised["grounded_claim_rate"]
    assert verdict.citation_coverage_rate == revised["citation_coverage_rate"]

    assert [entry["passed"] for entry in plain["draft_history"]] == [True]


async def test_the_result_file_pairs_the_loop_against_draft_one(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    release: BuiltRelease,
    scripted: Callable[[list[Turn]], ScriptedChat],
) -> None:
    """§14 Paired loop report (S6.4): a token-checked run's result file carries `paired_loop`,
    the report over its raw runs: the revised run improves on draft 1, the run that passed on
    draft 1 has delta 0, nothing regresses."""
    scripted([FIRST_DRAFT_FAILS, CLEAN, CLEAN])

    result = await run_one_ticker_twice(monkeypatch, release, workdir)

    report = result["paired_loop"]
    assert report == quality_baseline.paired_loop_report(result["raw_runs"])
    assert report["revised_runs"] == 1
    assert report["regressions"] == []
    by_key = {(entry["ticker"], entry["repeat"]): entry for entry in report["runs"]}
    assert len(by_key) == len(result["raw_runs"]) == 2
    for run in result["raw_runs"]:
        entry = by_key[(run["ticker"], run["repeat"])]
        assert entry["draft_count"] == run["draft_count"]
        assert entry["no_loop"] == run["draft_history"][0]["grounded_claim_rate"]
        assert entry["loop"] == run["grounded_claim_rate"]
        assert entry["delta"] == pytest.approx(entry["loop"] - entry["no_loop"])
        if run["draft_count"] > 1:
            assert entry["delta"] > 0
        else:
            assert entry["delta"] == 0


def raw(ticker: str, repeat: int, rates: list[float], final: float, **extra: Any) -> dict:
    return {
        "ticker": ticker,
        "repeat": repeat,
        "draft_count": len(rates),
        "draft_history": [
            {"grounded_claim_rate": rate, "passed": rate >= 0.75} for rate in rates
        ],
        "grounded_claim_rate": final,
        "recorded_memo": "ignored",
        **extra,
    }


def paired(ticker: str, repeat: int, count: int, no_loop, loop, delta) -> dict:
    return {
        "ticker": ticker,
        "repeat": repeat,
        "draft_count": count,
        "no_loop": no_loop,
        "loop": loop,
        "delta": delta,
    }


def test_paired_loop_report_lists_each_run_counts_revisions_and_names_regressions() -> None:
    """§14 Paired loop report (S6.4): per run the no-loop arm (draft 1), the loop arm (final), the
    delta and draft_count; revised_runs counts runs with more than one draft; `regressions` is
    every run whose loop arm is below its no-loop arm; a run that passed on draft 1 has delta 0
    and is no regression; a run without a draft history has no no-loop arm and is no regression."""
    runs = [
        raw("AAPL", 0, [1.0], 1.0),
        raw("AAPL", 1, [0.5, 0.75], 0.75),
        raw("MSFT", 0, [0.75, 0.5, 0.5], 0.5),
        raw("MSFT", 1, [], 0.5, draft_count=1),
    ]
    before = copy.deepcopy(runs)

    report = quality_baseline.paired_loop_report(runs)

    expected = [
        paired("AAPL", 0, 1, 1.0, 1.0, 0.0),
        paired("AAPL", 1, 2, 0.5, 0.75, 0.25),
        paired("MSFT", 0, 3, 0.75, 0.5, -0.25),
        paired("MSFT", 1, 1, None, 0.5, None),
    ]
    assert set(report) == {"runs", "revised_runs", "regressions"}
    assert sorted(report["runs"], key=lambda e: (e["ticker"], e["repeat"])) == expected
    assert report["revised_runs"] == 2
    assert report["regressions"] == [expected[2]]
    assert runs == before


def test_paired_loop_report_of_runs_that_never_regress_has_no_regressions() -> None:
    """§14 Paired loop report (S6.4): `regressions` is empty when no run's loop arm is below its
    no-loop arm; an equal arm after a revision is no regression either."""
    runs = [raw("AAPL", 0, [0.5, 0.5], 0.5), raw("AAPL", 1, [0.25, 1.0], 1.0)]

    report = quality_baseline.paired_loop_report(runs)

    assert report["regressions"] == []
    assert report["revised_runs"] == 2
    assert {entry["repeat"]: entry["delta"] for entry in report["runs"]} == {0: 0.0, 1: 0.75}


def test_paired_loop_report_of_no_runs_is_empty() -> None:
    """§14 Paired loop report (S6.4): no runs, no entries, nothing revised, no regressions."""
    assert quality_baseline.paired_loop_report([]) == {
        "runs": [],
        "revised_runs": 0,
        "regressions": [],
    }
