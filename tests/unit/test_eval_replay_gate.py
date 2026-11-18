"""eval-replay regression gate (docs/test-plan.md §14, S6.2).

``python -m evaluation.eval_replay --baseline <run_id>`` re-scores the drafts recorded in a pinned
approved baseline with the real runtime verifier and fails when a mean rate falls more than 2σ
below the baseline's recorded per-run rates.

Synthetic baselines are written with the real registry writer and a result file under the
throwaway working directory. Drafts are 20-claim memos whose cited sentences' figures are, or are
not, in the cited entry's text, so the real verifier yields known rates: an unsupported cited
claim lowers ``grounded_claim_rate`` only, an uncited claim lowers ``citation_coverage_rate`` only.
The recorded rates come from scoring the baseline drafts; a regression is simulated by recording
different drafts under the same recorded rates. ``evaluation.eval_replay`` is imported inside
``run_gate`` so a red run fails per test on the missing module.
"""

from __future__ import annotations

import json
import re
import socket
import statistics
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NoReturn

import pytest

from app.verification.grounding import evaluate_memo_grounding
from evaluation.registry import write_run

CLAIMS = 20
# (unsupported cited claims, uncited claims) per baseline run.
BASELINE_SHAPES: tuple[tuple[int, int], ...] = ((0, 0), (2, 2), (4, 4))
RUN_ID = "20261118T100000Z-abc1234-0001"
RELEASE_ID = "rel-synthetic-1"


def registry_entries() -> list[dict[str, Any]]:
    return [
        {
            "index": i + 1,
            "source_type": "news",
            "title": f"Segment {i} report",
            "text": f"Segment {i} revenue was ${100 + i} billion in fiscal 2024.",
            "url": f"https://news.example.test/{i}",
            "date": "2026-06-30",
        }
        for i in range(CLAIMS)
    ]


def memo_for(unsupported: int, uncited: int) -> str:
    """A memo of CLAIMS sentences: `unsupported` cite a wrong figure, `uncited` carry no citation."""
    sentences = []
    for i in range(CLAIMS):
        if i < unsupported:
            sentences.append(
                f"Segment {i} revenue was ${900 + i} billion in fiscal 2024 [{i + 1}]."
            )
        elif i < unsupported + uncited:
            sentences.append(f"Segment {i} revenue was ${100 + i} billion in fiscal 2024.")
        else:
            sentences.append(
                f"Segment {i} revenue was ${100 + i} billion in fiscal 2024 [{i + 1}]."
            )
    return "\n".join(sentences)


def score(shape: tuple[int, int]) -> tuple[float, float]:
    verdict = evaluate_memo_grounding(memo_for(*shape), registry_entries())
    return verdict.grounded_claim_rate, verdict.citation_coverage_rate


def baseline_rates(index: int) -> list[float]:
    return [score(shape)[index] for shape in BASELINE_SHAPES]


def sigma_below(
    index: int, shapes: Sequence[tuple[int, int]], *, population: bool = False
) -> float:
    """How many σ of the baseline's recorded rates the candidate's mean lies below its mean."""
    base = baseline_rates(index)
    spread = statistics.pstdev(base) if population else statistics.stdev(base)
    candidate = statistics.mean(score(shape)[index] for shape in shapes)
    return (statistics.mean(base) - candidate) / spread


@pytest.fixture(autouse=True)
def socket_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Any non-AF_UNIX connect raises and is recorded: the gate is offline."""
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(address: object) -> NoReturn:
        attempts.append(repr(address))
        raise OSError(f"network access attempted by the eval-replay gate: {address!r}")

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
            return
        refuse(address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family == socket.AF_UNIX:
            return real_connect_ex(self, address)
        refuse(address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        refuse(address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    yield attempts
    assert attempts == [], f"network connect attempted during test: {attempts}"


def write_baseline(
    workdir: Path,
    *,
    recorded_shapes: Sequence[tuple[int, int]] = BASELINE_SHAPES,
    rescored_shapes: Sequence[tuple[int, int]] | None = None,
    run_id: str = RUN_ID,
    classification: str = "approved",
    evidence_release: str = RELEASE_ID,
    checker: str = "token",
    registry_root: Path | None = None,
) -> Path:
    """Write a baseline's result file and registry record; return the registry root.

    The per-run rates are those of `recorded_shapes`; the drafts stored in the result file are
    those of `rescored_shapes` (default: the same), so a regression is a draft that now scores
    lower than its recorded rate.
    """
    drafts = list(rescored_shapes if rescored_shapes is not None else recorded_shapes)
    raw_runs: list[dict[str, Any]] = []
    for number, (recorded, draft) in enumerate(zip(recorded_shapes, drafts, strict=True)):
        grounded, coverage = score(recorded)
        raw_runs.append(
            {
                "ticker": ("AAPL", "MSFT", "NVDA")[number % 3],
                "grounded_claim_rate": grounded,
                "citation_coverage_rate": coverage,
                "recorded_memo": memo_for(*draft),
                "recorded_registry": registry_entries(),
            }
        )
    lineage: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": "abc1234",
        "branch": "main",
        "model": "fake-model",
        "temperature": 0.0,
        "prompt_version": "p" * 16,
        "evidence_release": evidence_release,
        "snapshot_hash": "a" * 64,
        "fixture_version": "b" * 16,
    }
    result_path = f"evaluation/quality_res/{run_id}.json"
    result_file = workdir / result_path
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps({**lineage, "grounding_checker": checker, "raw_runs": raw_runs})
    )
    record = {
        **lineage,
        "metrics": {
            "grounded_claim_rate": statistics.mean(r["grounded_claim_rate"] for r in raw_runs),
            "citation_coverage": statistics.mean(r["citation_coverage_rate"] for r in raw_runs),
        },
        "classification": classification,
        "result_path": result_path,
    }
    root = registry_root if registry_root is not None else workdir / "registry" / "runs"
    if evidence_release == "live" and classification == "approved":
        # The writer refuses this record; the gate must still refuse a hand-placed one.
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{run_id}.json").write_text(json.dumps(record))
    else:
        write_run(record, root=root)
    return root


def run_gate(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[Any, str]:
    """Run the gate; return its exit code (return value or SystemExit code) and its output."""
    from evaluation import eval_replay

    try:
        code: Any = eval_replay.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, f"{captured.out}\n{captured.err}"


def floats_in(output: str) -> list[float]:
    return [float(token) for token in re.findall(r"\d+\.\d+", output)]


def assert_prints(output: str, *values: float) -> None:
    """Each value (or its magnitude) appears in the output as a decimal number, any precision."""
    found = floats_in(output)
    for value in values:
        assert any(abs(token - abs(value)) < 0.0051 for token in found), (
            f"{value:.4f} not printed; numbers in output: {found}"
        )


# --- Regression gate -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rescored",
    [BASELINE_SHAPES, ((0, 0),) * 3],
    ids=["drafts-rescore-at-recorded-rates", "drafts-improved"],
)
def test_drafts_at_or_above_recorded_rates_pass_and_output_shows_means_and_sigma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rescored: Sequence[tuple[int, int]],
) -> None:
    """§14 Regression gate (S6.2): drafts re-scoring at the recorded rates pass; prints means, σ."""
    monkeypatch.chdir(tmp_path)
    root = write_baseline(tmp_path, rescored_shapes=rescored)

    code, output = run_gate(["--baseline", RUN_ID, "--registry-root", str(root)], capsys)

    assert code == 0, output
    for index in (0, 1):
        base = baseline_rates(index)
        assert_prints(output, statistics.mean(base), statistics.stdev(base))


def test_grounded_claim_rate_2_5_sigma_below_baseline_fails_naming_the_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Regression gate (S6.2): grounded_claim_rate ~2.5σ below the baseline fails the gate."""
    monkeypatch.chdir(tmp_path)
    regressed = ((9, 0),) * 3
    assert 2.5 < sigma_below(0, regressed) < 2.8
    assert sigma_below(1, regressed) < 0  # citation coverage did not move down
    root = write_baseline(tmp_path, rescored_shapes=regressed)

    code, output = run_gate(["--baseline", RUN_ID, "--registry-root", str(root)], capsys)

    assert code == 1, output
    assert "grounded_claim_rate" in output
    base = baseline_rates(0)
    assert_prints(output, statistics.mean(base), statistics.stdev(base))
    candidate = score((9, 0))[0]
    assert_prints(output, *(candidate - recorded for recorded in base))


def test_citation_coverage_2_5_sigma_below_baseline_fails_naming_the_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Regression gate (S6.2): citation_coverage ~2.5σ below the baseline fails the gate."""
    monkeypatch.chdir(tmp_path)
    regressed = ((0, 7),) * 3
    assert 2.49 < sigma_below(1, regressed) < 2.51
    assert sigma_below(0, regressed) < 0  # grounded rate did not move down
    root = write_baseline(tmp_path, rescored_shapes=regressed)

    code, output = run_gate(["--baseline", RUN_ID, "--registry-root", str(root)], capsys)

    assert code == 1, output
    assert "citation_coverage" in output
    base = baseline_rates(1)
    assert_prints(output, statistics.mean(base), statistics.stdev(base))
    candidate = score((0, 7))[1]
    assert_prints(output, *(candidate - recorded for recorded in base))


@pytest.mark.parametrize(
    ("rescored", "metric_index"),
    [
        (((6, 0),) * 3, 0),
        (((0, 5),) * 3, 1),
        (((0, 6), (0, 6), (0, 5)), 1),
    ],
    ids=[
        "grounded-1.4-sigma",
        "coverage-1.5-sigma",
        "coverage-between-sample-and-population-2-sigma",
    ],
)
def test_a_drop_inside_two_sample_sigma_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rescored: Sequence[tuple[int, int]],
    metric_index: int,
) -> None:
    """§14 Regression gate (S6.2): a mean within 2σ of the baseline mean passes.

    σ is the sample standard deviation: the last case lies inside 2 sample σ but outside 2
    population σ, so a gate using the population deviation would fail it.
    """
    monkeypatch.chdir(tmp_path)
    assert 0 < sigma_below(metric_index, rescored) < 2.0
    if rescored == ((0, 6), (0, 6), (0, 5)):
        assert sigma_below(metric_index, rescored, population=True) > 2.0
    root = write_baseline(tmp_path, rescored_shapes=rescored)

    code, output = run_gate(["--baseline", RUN_ID, "--registry-root", str(root)], capsys)

    assert code == 0, output


def test_the_gate_reads_the_default_registry_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Regression gate (S6.2): without --registry-root the record is read from evaluation/registry/runs."""
    monkeypatch.chdir(tmp_path)
    write_baseline(tmp_path, registry_root=tmp_path / "evaluation" / "registry" / "runs")

    code, output = run_gate(["--baseline", RUN_ID], capsys)

    assert code == 0, output


# --- Pinned approved baseline ----------------------------------------------------------------


REFUSALS: list[tuple[list[str], dict[str, Any], tuple[str, ...]]] = [
    (["--registry-root", "{root}"], {}, ("baseline",)),
    (["--baseline", "no-such-run", "--registry-root", "{root}"], {}, ("no-such-run",)),
    (
        ["--baseline", RUN_ID, "--registry-root", "{root}"],
        {"classification": "candidate"},
        ("approved", "candidate"),
    ),
    (
        ["--baseline", RUN_ID, "--registry-root", "{root}"],
        {"evidence_release": "live", "classification": "approved"},
        ("live",),
    ),
    (
        ["--baseline", RUN_ID, "--registry-root", "{root}"],
        {"evidence_release": "live", "classification": "candidate"},
        ("live", "approved"),
    ),
    (["--baseline", RUN_ID, "--registry-root", "{root}"], {"checker": "semantic"}, ("token", "semantic", "checker")),
]


@pytest.mark.parametrize(
    ("argv", "baseline", "reason"),
    REFUSALS,
    ids=[
        "missing-baseline-flag",
        "unknown-run-id",
        "candidate-record",
        "live-approved-record-placed-by-hand",
        "live-candidate-record",
        "result-scored-with-semantic-checker",
    ],
)
def test_gate_refuses_a_baseline_that_is_not_a_pinned_approved_token_replay_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    baseline: dict[str, Any],
    reason: tuple[str, ...],
) -> None:
    """§14 Pinned approved baseline (S6.2): exit 1 before scoring, saying why.

    Every baseline here holds drafts that would pass the gate, so exit 1 can only be the refusal.
    """
    monkeypatch.chdir(tmp_path)
    root = write_baseline(tmp_path, **baseline)

    code, output = run_gate([arg.replace("{root}", str(root)) for arg in argv], capsys)

    assert code == 1, output
    assert any(word in output.lower() for word in reason), output


# --- Gate is offline -------------------------------------------------------------------------


def test_gate_passes_under_a_socket_guard_without_a_connection_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    socket_guard: list[str],
) -> None:
    """§14 Gate is offline (S6.2): a passing gate run opens no network connection."""
    monkeypatch.chdir(tmp_path)
    root = write_baseline(tmp_path)

    code, output = run_gate(["--baseline", RUN_ID, "--registry-root", str(root)], capsys)

    assert code == 0, output
    assert socket_guard == []
