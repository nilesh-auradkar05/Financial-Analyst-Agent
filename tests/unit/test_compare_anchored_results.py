"""S2-T05: paired comparator for anchored benchmark result files.

Oracle: docs/test-plan.md §6 "Paired comparator"; docs/retrieval-benchmark.md
§8 (paired comparison policy), §9a and §10 (embedding identifiers).

The anchored result dicts are built here with the shape the S2-T04 runner
emits. They are test inputs, not fakes of a system.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evaluation.compare_retrieval_results import (
    ANCHORED_METRICS,
    COMPARISON_AXES,
    ComparisonError,
    compare_anchored_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KS = ("5", "10")
LATENCY_METRICS = ("latency_cold_ms", "latency_warm_ms")
DEFAULT_QUALITY = {
    "pass": 1.0,
    "recall": 0.5,
    "precision": 0.5,
    "mrr": 0.5,
    "ndcg": 0.5,
    "section_recall": 0.5,
    "keyword_hit_rate": 0.5,
}


def _case(
    case_id: str,
    *,
    rank: int | None = 1,
    cold: float = 100.0,
    warm: float = 50.0,
    **quality: float | None,
) -> dict[str, Any]:
    """One per-case row. Ranks used are <= 5 so every K agrees on hit or miss."""
    values = {**DEFAULT_QUALITY, **quality}
    return {
        "case_id": case_id,
        "first_relevant_rank": rank,
        "metrics": {k: {**values, "first_relevant_rank": rank} for k in KS},
        "latency_ms": {"cold": cold, "warm": warm},
    }


def _result(
    cases: list[dict[str, Any]],
    *,
    backend: str = "chroma",
    method: str = "dense",
    embedding_model: str = "qwen3-embedding:4b",
    fixture_version: str = "v1",
    mode: str | None = None,
) -> dict[str, Any]:
    return {
        "fixture_version": fixture_version,
        "backend": backend,
        "retrieval_method": method,
        "mode": method if mode is None else mode,
        "ks": [5, 10],
        "top_k": 10,
        "provenance": {"embedding_model": embedding_model},
        "cases": cases,
        "summary": {},
    }


def _ids(*ids: str) -> list[dict[str, Any]]:
    return [_case(i) for i in ids]


def _metric(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    (row,) = [r for r in rows if r["metric"] == name]
    return row


def _quality(report: dict[str, Any], k: str, name: str) -> dict[str, Any]:
    return _metric(report["quality"][k], name)


def _latency(report: dict[str, Any], name: str) -> dict[str, Any]:
    return _metric(report["latency"], name)


# --- public surface ---------------------------------------------------------


def test_public_constants_and_error_type() -> None:
    """test-plan §6 Paired comparator: Output (public surface)."""
    assert COMPARISON_AXES == ("backend", "retrieval_method", "embedding_model")
    assert ANCHORED_METRICS == (
        "pass",
        "recall",
        "precision",
        "mrr",
        "ndcg",
        "section_recall",
        "keyword_hit_rate",
        "miss_rate",
    )
    assert issubclass(ComparisonError, ValueError)


# --- Mismatched fixture_version --------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
def test_mismatched_fixture_version_is_rejected(strict: bool) -> None:
    """test-plan §6 Paired comparator: Mismatched `fixture_version` -> rejected."""
    base = _result(_ids("c1", "c2"), fixture_version="v1")
    cand = _result(_ids("c1", "c2"), fixture_version="v2")
    with pytest.raises(ComparisonError, match="fixture_version"):
        compare_anchored_results(base, cand, strict_case_ids=strict)


# --- Single-axis rule -------------------------------------------------------

_BACKEND = {"backend": "qdrant"}
_METHOD = {"method": "hybrid"}
_EMBED = {"embedding_model": "text-embedding-3-small"}


@pytest.mark.parametrize(
    ("changed", "differing"),
    [
        ({**_BACKEND, **_METHOD}, ("backend", "retrieval_method")),
        ({**_BACKEND, **_EMBED}, ("backend", "embedding_model")),
        ({**_METHOD, **_EMBED}, ("retrieval_method", "embedding_model")),
        ({**_BACKEND, **_METHOD, **_EMBED}, COMPARISON_AXES),
    ],
)
def test_two_or_more_differing_axes_are_rejected(
    changed: dict[str, str], differing: tuple[str, ...]
) -> None:
    """test-plan §6 Paired comparator: Both backend and method differ -> rejected,
    single-axis rule (retrieval-benchmark §8; embedding identifiers §9a/§10)."""
    base = _result(_ids("c1", "c2"))
    cand = _result(_ids("c1", "c2"), **changed)
    with pytest.raises(ComparisonError) as excinfo:
        compare_anchored_results(base, cand)
    for axis in differing:
        assert axis in str(excinfo.value)


@pytest.mark.parametrize(
    ("changed", "axis"),
    [
        (_BACKEND, "backend"),
        (_METHOD, "retrieval_method"),
        (_EMBED, "embedding_model"),
        ({}, None),
    ],
)
def test_at_most_one_differing_axis_is_accepted_and_named(
    changed: dict[str, str], axis: str | None
) -> None:
    """test-plan §6 Paired comparator: Both backend and method differ -> rejected
    (boundary: exactly one or zero differing axes is accepted, `axis` names it)."""
    base = _result(_ids("c1", "c2"))
    cand = _result(_ids("c1", "c2"), **changed)
    report = compare_anchored_results(base, cand)
    assert report["axis"] == axis
    assert report["shared_case_count"] == 2
    assert report["fixture_version"] == "v1"
    assert report["baseline"] == {
        "backend": "chroma",
        "retrieval_method": "dense",
        "embedding_model": "qwen3-embedding:4b",
    }
    assert report["candidate"] == {
        "backend": changed.get("backend", "chroma"),
        "retrieval_method": changed.get("method", "dense"),
        "embedding_model": changed.get("embedding_model", "qwen3-embedding:4b"),
    }


# --- Strict case ids --------------------------------------------------------


def test_missing_case_under_strict_case_ids_fails_naming_both_sides() -> None:
    """test-plan §6 Paired comparator: Missing case under `--strict-case-ids` -> fails."""
    base = _result(_ids("c1", "c2", "c3"))
    cand = _result(_ids("c1", "c2", "c4"))
    with pytest.raises(ComparisonError) as excinfo:
        compare_anchored_results(base, cand, strict_case_ids=True)
    assert "c3" in str(excinfo.value)
    assert "c4" in str(excinfo.value)


def test_identical_case_sets_pass_under_strict_case_ids() -> None:
    """test-plan §6 Paired comparator: Missing case under `--strict-case-ids`
    (boundary: equal sets are not a violation, order is irrelevant)."""
    base = _result(_ids("c1", "c2", "c3"))
    cand = _result(_ids("c3", "c1", "c2"))
    report = compare_anchored_results(base, cand, strict_case_ids=True)
    assert report["shared_case_count"] == 3
    assert report["baseline_only_cases"] == []
    assert report["candidate_only_cases"] == []


def test_without_strict_flag_unshared_cases_are_reported_not_fatal() -> None:
    """test-plan §6 Paired comparator: Missing case under `--strict-case-ids`
    (non-strict side: compare shared ids and report the rest)."""
    base = _result(_ids("c1", "c2", "c3"))
    cand = _result(_ids("c1", "c2", "c4"))
    report = compare_anchored_results(base, cand)
    assert report["shared_case_count"] == 2
    assert report["baseline_only_cases"] == ["c3"]
    assert report["candidate_only_cases"] == ["c4"]
    assert _quality(report, "5", "recall")["shared_count"] == 2


# --- Mode / method held constant -------------------------------------------


@pytest.mark.parametrize("mislabelled", ["baseline", "candidate"])
def test_mode_not_matching_retrieval_method_fails(mislabelled: str) -> None:
    """test-plan §6 Paired comparator: Mismatched mode/method when held constant -> fails."""
    good = _result(_ids("c1", "c2"))
    bad = _result(_ids("c1", "c2"), mode="hybrid")
    base, cand = (bad, good) if mislabelled == "baseline" else (good, bad)
    with pytest.raises(ComparisonError, match="mode"):
        compare_anchored_results(base, cand)


# --- Null first relevant rank counted as miss ------------------------------


def test_null_first_relevant_rank_counts_as_miss() -> None:
    """test-plan §6 Paired comparator: Null first relevant rank -> counted as miss."""
    base = _result(
        [
            _case("c1", rank=1),
            _case("c2", rank=1),
            _case("c3", rank=1),
            _case("c4", rank=None),
        ]
    )
    cand = _result(
        [
            _case("c1", rank=None),
            _case("c2", rank=1),
            _case("c3", rank=1),
            _case("c4", rank=None),
        ]
    )
    report = compare_anchored_results(base, cand, bootstrap_samples=500)
    for k in KS:
        miss = _quality(report, k, "miss_rate")
        assert miss["baseline_mean"] == pytest.approx(0.25)
        assert miss["candidate_mean"] == pytest.approx(0.5)
        assert miss["paired_delta"] == pytest.approx(0.25)
        # lower miss_rate is better: one hit turned into a miss is the only loss
        assert (miss["wins"], miss["ties"], miss["losses"]) == (0, 3, 1)


# --- Output ----------------------------------------------------------------

_BASE_RECALL = [0.2, 0.4, 0.6, 0.8]
_CAND_RECALL = [0.4, 0.4, 0.5, 1.0]
_BASE_COLD = [100.0, 200.0, 300.0, 400.0]
_CAND_COLD = [90.0, 210.0, 300.0, 350.0]
_BASE_WARM = [10.0, 20.0, 30.0, 40.0]
_CAND_WARM = [10.0, 25.0, 20.0, 40.0]
_NAMES = ["c1", "c2", "c3", "c4"]


def _hand_computed() -> tuple[dict[str, Any], dict[str, Any]]:
    base = _result(
        [
            _case(n, recall=r, cold=c, warm=w)
            for n, r, c, w in zip(_NAMES, _BASE_RECALL, _BASE_COLD, _BASE_WARM)
        ]
    )
    cand = _result(
        [
            _case(n, recall=r, cold=c, warm=w)
            for n, r, c, w in zip(_NAMES, _CAND_RECALL, _CAND_COLD, _CAND_WARM)
        ]
    )
    return base, cand


def test_output_reports_every_metric_for_every_k() -> None:
    """test-plan §6 Paired comparator: Output -> paired deltas for each K and metric."""
    base, cand = _hand_computed()
    report = compare_anchored_results(base, cand, bootstrap_samples=500)
    assert set(report["quality"]) == set(KS)
    for k in KS:
        assert [r["metric"] for r in report["quality"][k]] == list(ANCHORED_METRICS)


@pytest.mark.parametrize("k", KS)
def test_output_paired_delta_relative_delta_and_win_tie_loss(k: str) -> None:
    """test-plan §6 Paired comparator: Output -> paired deltas, win/tie/loss.

    Recall diffs are [+0.2, 0.0, -0.1, +0.2]: mean 0.075, baseline mean 0.5.
    """
    base, cand = _hand_computed()
    row = _quality(compare_anchored_results(base, cand, bootstrap_samples=500), k, "recall")
    assert row["baseline_mean"] == pytest.approx(0.5)
    assert row["candidate_mean"] == pytest.approx(0.575)
    assert row["paired_delta"] == pytest.approx(0.075)
    assert row["relative_delta"] == pytest.approx(0.15)
    assert (row["wins"], row["ties"], row["losses"]) == (2, 1, 1)
    assert row["shared_count"] == 4


def test_output_bootstrap_ci_brackets_the_paired_delta() -> None:
    """test-plan §6 Paired comparator: Output -> bootstrap CI on the paired delta."""
    base, cand = _hand_computed()
    report = compare_anchored_results(base, cand, bootstrap_samples=2000)
    row = _quality(report, "5", "recall")
    assert row["ci_low"] <= row["paired_delta"] <= row["ci_high"]
    # a resampled mean of the diffs can never leave their observed range
    assert -0.1 - 1e-9 <= row["ci_low"]
    assert row["ci_high"] <= 0.2 + 1e-9
    assert row["ci_low"] < row["ci_high"]


def test_output_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    """test-plan §6 Paired comparator: Output -> bootstrap CI (seeded, reproducible)."""
    base, cand = _hand_computed()
    first = compare_anchored_results(base, cand, bootstrap_samples=1000, seed=7)
    second = compare_anchored_results(base, cand, bootstrap_samples=1000, seed=7)
    assert first == second
    assert first["bootstrap_samples"] == 1000


def test_output_default_bootstrap_samples_meet_the_policy_floor() -> None:
    """test-plan §6 Paired comparator: Output; retrieval-benchmark §8 (>= 10k resamples)."""
    base, cand = _hand_computed()
    report = compare_anchored_results(base, cand)
    assert report["bootstrap_samples"] >= 10_000


def test_output_report_is_json_serialisable() -> None:
    """test-plan §6 Paired comparator: Output (machine-readable report)."""
    base, cand = _hand_computed()
    report = compare_anchored_results(base, cand, bootstrap_samples=200)
    assert json.loads(json.dumps(report)) == report


def test_output_self_compare_has_zero_deltas_and_all_ties() -> None:
    """test-plan §6 Paired comparator: Output (self-compare is all ties, CI [0, 0])."""
    base, _ = _hand_computed()
    report = compare_anchored_results(base, base, bootstrap_samples=500)
    assert report["axis"] is None
    rows = [r for k in KS for r in report["quality"][k]] + report["latency"]
    assert rows
    for row in rows:
        assert row["paired_delta"] == pytest.approx(0.0)
        assert (row["wins"], row["losses"]) == (0, 0)
        assert row["ties"] == 4
        assert row["ci_low"] == pytest.approx(0.0)
        assert row["ci_high"] == pytest.approx(0.0)
        assert row["baseline_mean"] == pytest.approx(row["candidate_mean"])


def test_output_latency_is_separated_and_lower_is_better() -> None:
    """test-plan §6 Paired comparator: Output -> separated latency.

    Cold diffs [-10, +10, 0, -50]: mean -12.5 over baseline mean 250.
    Warm diffs [0, +5, -10, 0]: mean -1.25 over baseline mean 25.
    """
    base, cand = _hand_computed()
    report = compare_anchored_results(base, cand, bootstrap_samples=500)

    assert [r["metric"] for r in report["latency"]] == list(LATENCY_METRICS)
    for k in KS:
        assert not any("latency" in r["metric"] for r in report["quality"][k])

    cold = _latency(report, "latency_cold_ms")
    assert cold["baseline_mean"] == pytest.approx(250.0)
    assert cold["candidate_mean"] == pytest.approx(237.5)
    assert cold["paired_delta"] == pytest.approx(-12.5)
    assert cold["relative_delta"] == pytest.approx(-0.05)
    assert (cold["wins"], cold["ties"], cold["losses"]) == (2, 1, 1)

    warm = _latency(report, "latency_warm_ms")
    assert warm["baseline_mean"] == pytest.approx(25.0)
    assert warm["paired_delta"] == pytest.approx(-1.25)
    assert (warm["wins"], warm["ties"], warm["losses"]) == (1, 2, 1)


def test_output_null_keyword_hit_rate_pair_is_skipped_for_that_metric_only() -> None:
    """test-plan §6 Paired comparator: Output (a null on either side drops the pair
    from keyword_hit_rate only)."""
    base = _result(
        [
            _case("c1", keyword_hit_rate=0.5),
            _case("c2", keyword_hit_rate=None),
            _case("c3", keyword_hit_rate=0.5),
        ]
    )
    cand = _result(
        [
            _case("c1", keyword_hit_rate=0.7),
            _case("c2", keyword_hit_rate=0.5),
            _case("c3", keyword_hit_rate=None),
        ]
    )
    report = compare_anchored_results(base, cand, bootstrap_samples=200)
    for k in KS:
        kw = _quality(report, k, "keyword_hit_rate")
        assert kw["shared_count"] == 1
        assert kw["paired_delta"] == pytest.approx(0.2)
        assert (kw["wins"], kw["ties"], kw["losses"]) == (1, 0, 0)
        assert _quality(report, k, "recall")["shared_count"] == 3
        assert _quality(report, k, "miss_rate")["shared_count"] == 3


def test_output_relative_delta_is_none_when_baseline_mean_is_zero() -> None:
    """test-plan §6 Paired comparator: Output (relative delta undefined at a zero baseline)."""
    base = _result([_case("c1", recall=0.0), _case("c2", recall=0.0)])
    cand = _result([_case("c1", recall=0.5), _case("c2", recall=0.5)])
    row = _quality(compare_anchored_results(base, cand, bootstrap_samples=200), "5", "recall")
    assert row["paired_delta"] == pytest.approx(0.5)
    assert row["relative_delta"] is None


# --- CLI -------------------------------------------------------------------


def _run_cli(tmp_path: Path, base: dict[str, Any], cand: dict[str, Any], *flags: str):
    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    cand_path.write_text(json.dumps(cand), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.compare_retrieval_results",
            str(base_path),
            str(cand_path),
            *flags,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_strict_case_ids_violation_exits_nonzero_naming_the_case(tmp_path: Path) -> None:
    """test-plan §6 Paired comparator: Missing case under `--strict-case-ids` -> fails (CLI)."""
    base = _result(_ids("c1", "c2", "missing-case"))
    cand = _result(_ids("c1", "c2"))
    proc = _run_cli(tmp_path, base, cand, "--strict-case-ids")
    assert proc.returncode != 0
    assert "missing-case" in proc.stdout + proc.stderr


def test_cli_anchored_compare_succeeds_and_writes_json_report(tmp_path: Path) -> None:
    """test-plan §6 Paired comparator: Output (CLI prints a report and writes the JSON)."""
    base, cand = _hand_computed()
    out = tmp_path / "report.json"
    proc = _run_cli(
        tmp_path,
        base,
        cand,
        "--strict-case-ids",
        "--bootstrap-samples",
        "200",
        "--json-output",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert "latency" in proc.stdout.lower()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["shared_case_count"] == 4
    assert report["bootstrap_samples"] == 200
    assert _quality(report, "5", "recall")["paired_delta"] == pytest.approx(0.075)
