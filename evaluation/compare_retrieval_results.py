"""Compare retrieval evaluation outputs with paired case-id methodology.

This comparator intentionally avoids the earlier mistake of averaging each file
independently. Method comparisons are meaningful only when the same case IDs are
present on both sides. The script therefore joins rows by ``case_id``, reports
baseline-only/candidate-only cases, and computes paired deltas over shared cases.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HIGHER_IS_BETTER = {
    "precision_at_5",
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "keyword_hit_rate",
    "section_recall_at_k",
    "section_hit_at_k",
    "pass_at_k",
}

LOWER_IS_BETTER = {
    "latency_ms",
    "miss_rate",
    "first_relevant_rank_mean_among_hits",
}

CONFIG_FIELDS = {
    "top_k",
    "retrieved_count",
}

DEFAULT_METRICS = [
    "precision_at_5",
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "keyword_hit_rate",
    "section_recall_at_k",
    "miss_rate",
    "first_relevant_rank_mean_among_hits",
]

DEFAULT_RUNTIME_METRICS = [
    "latency_ms",
    "latency_ms_warm",
]

FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    """Normalized retrieval-result row."""

    case_id: str
    mode: str | None
    metrics: dict[str, Any]
    retrieval_methods: frozenset[str]
    raw: dict[str, Any]
    index: int


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Paired metric comparison for one metric."""

    metric: str
    baseline_mean: float | None
    candidate_mean: float | None
    paired_delta: float | None
    relative_delta: float | None
    wins: int
    ties: int
    losses: int
    ci_low: float | None
    ci_high: float | None
    shared_count: int


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load a retrieval-eval JSON payload."""

    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {payload_path}")
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"Expected payload.results list in {payload_path}")
    return payload


def rows_by_case_id(payload: Mapping[str, Any]) -> dict[str, RetrievalRow]:
    """Normalize result rows and return them keyed by case_id."""

    rows: dict[str, RetrievalRow] = {}
    duplicate_case_ids: list[str] = []

    for index, item in enumerate(payload.get("results", [])):
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id in rows:
            duplicate_case_ids.append(case_id)
            continue
        rows[case_id] = RetrievalRow(
            case_id=case_id,
            mode=_normalize_optional_string(item.get("mode")),
            metrics=dict(item.get("metrics") or {}),
            retrieval_methods=_extract_retrieval_methods(item),
            raw=item,
            index=index,
        )

    if duplicate_case_ids:
        duplicates = ", ".join(sorted(set(duplicate_case_ids)))
        raise ValueError(f"Duplicate case_id values found: {duplicates}")

    return rows


def compare_payloads(
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    bootstrap_samples: int = 2_000,
    random_seed: int = 13,
    include_config: bool = False,
) -> dict[str, Any]:
    """Compare two payloads by joining rows on case_id."""

    baseline_rows = rows_by_case_id(baseline_payload)
    candidate_rows = rows_by_case_id(candidate_payload)

    baseline_ids = set(baseline_rows)
    candidate_ids = set(candidate_rows)
    shared_case_ids = [case_id for case_id in baseline_rows if case_id in candidate_rows]

    quality_metrics = _ordered_available_metrics(
        baseline_rows,
        candidate_rows,
        DEFAULT_METRICS,
        include_config=include_config,
    )
    runtime_metrics = _ordered_available_runtime_metrics(baseline_rows, candidate_rows)

    quality = [
        _compare_metric(
            metric,
            baseline_rows,
            candidate_rows,
            shared_case_ids,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        for metric in quality_metrics
    ]
    runtime = [
        _compare_metric(
            metric,
            baseline_rows,
            candidate_rows,
            shared_case_ids,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        for metric in runtime_metrics
    ]

    return {
        "shared_case_count": len(shared_case_ids),
        "baseline_case_count": len(baseline_rows),
        "candidate_case_count": len(candidate_rows),
        "shared_case_ids": shared_case_ids,
        "baseline_only_cases": sorted(baseline_ids - candidate_ids),
        "candidate_only_cases": sorted(candidate_ids - baseline_ids),
        "quality": [asdict(comparison) for comparison in quality],
        "runtime": [asdict(comparison) for comparison in runtime],
    }


def validate_payload_contract(
    payload: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_method: str | None = None,
    label: str,
) -> None:
    """Validate result mode/method so mislabeled files fail loudly."""

    rows = rows_by_case_id(payload)
    if expected_mode:
        bad_modes = [
            f"{row.case_id}: {row.mode!r}"
            for row in rows.values()
            if row.mode != expected_mode
        ]
        if bad_modes:
            details = "; ".join(bad_modes[:10])
            raise ValueError(
                f"{label} expected mode={expected_mode!r}, but found mismatches: {details}"
            )

    if expected_method:
        bad_methods: list[str] = []
        missing_methods: list[str] = []
        for row in rows.values():
            if not row.retrieval_methods:
                missing_methods.append(row.case_id)
                continue
            if expected_method not in row.retrieval_methods:
                bad_methods.append(
                    f"{row.case_id}: {sorted(row.retrieval_methods)!r}"
                )
        if bad_methods or missing_methods:
            parts: list[str] = []
            if bad_methods:
                parts.append("mismatched methods: " + "; ".join(bad_methods[:10]))
            if missing_methods:
                parts.append("missing methods: " + ", ".join(missing_methods[:10]))
            raise ValueError(
                f"{label} expected retrieval_method={expected_method!r}, "
                + " | ".join(parts)
            )


def print_comparison_report(
    comparison: Mapping[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> None:
    """Print a readable comparison report."""

    print(f"Baseline:  {baseline_label}")
    print(f"Candidate: {candidate_label}")
    print()
    print(
        "cases | baseline | candidate | shared | baseline_only | candidate_only"
    )
    print("------|----------|-----------|--------|---------------|---------------")
    print(
        f"count | {comparison['baseline_case_count']} | "
        f"{comparison['candidate_case_count']} | "
        f"{comparison['shared_case_count']} | "
        f"{len(comparison['baseline_only_cases'])} | "
        f"{len(comparison['candidate_only_cases'])}"
    )

    _print_missing_cases("Baseline-only cases", comparison["baseline_only_cases"])
    _print_missing_cases("Candidate-only cases", comparison["candidate_only_cases"])

    if not comparison["shared_case_ids"]:
        print("\nNo shared case_id values. Refusing to compare means over different fixtures.")
        return

    print("\nQuality metrics, paired by case_id")
    _print_metric_table(comparison["quality"])

    print("\nRuntime metrics, paired by case_id")
    _print_metric_table(comparison["runtime"])


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_retrieval_methods(row: Mapping[str, Any]) -> frozenset[str]:
    methods: set[str] = set()
    direct = _normalize_optional_string(row.get("retrieval_method"))
    if direct:
        methods.add(direct)

    for packet in row.get("evidence_packets") or []:
        if not isinstance(packet, dict):
            continue
        packet_method = _normalize_optional_string(packet.get("retrieval_method"))
        if packet_method:
            methods.add(packet_method)
        metadata = packet.get("metadata")
        if isinstance(metadata, dict):
            metadata_method = _normalize_optional_string(metadata.get("retrieval_method"))
            if metadata_method:
                methods.add(metadata_method)

    return frozenset(methods)


def _metric_value(row: RetrievalRow, metric: str) -> float | None:
    if metric == "miss_rate":
        return 1.0 if row.metrics.get("first_relevant_rank") is None else 0.0
    if metric == "first_relevant_rank_mean_among_hits":
        value = row.metrics.get("first_relevant_rank")
        return _as_float(value) if value is not None else None
    if metric == "latency_ms_warm":
        return _as_float(row.metrics.get("latency_ms"))
    return _as_float(row.metrics.get(metric))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ordered_available_metrics(
    baseline_rows: Mapping[str, RetrievalRow],
    candidate_rows: Mapping[str, RetrievalRow],
    preferred: Sequence[str],
    *,
    include_config: bool,
) -> list[str]:
    available = _available_metric_names(baseline_rows, candidate_rows)
    available.update({"miss_rate", "first_relevant_rank_mean_among_hits"})
    if not include_config:
        available -= CONFIG_FIELDS
    runtime = {"latency_ms"}
    ordered = [metric for metric in preferred if metric in available and metric not in runtime]
    extras = sorted(
        metric
        for metric in available
        if metric not in set(ordered) | runtime | CONFIG_FIELDS
        and not metric.startswith("avg_")
    )
    return ordered + extras


def _ordered_available_runtime_metrics(
    baseline_rows: Mapping[str, RetrievalRow],
    candidate_rows: Mapping[str, RetrievalRow],
) -> list[str]:
    available = _available_metric_names(baseline_rows, candidate_rows)
    if "latency_ms" not in available:
        return []
    return DEFAULT_RUNTIME_METRICS


def _available_metric_names(
    baseline_rows: Mapping[str, RetrievalRow],
    candidate_rows: Mapping[str, RetrievalRow],
) -> set[str]:
    names: set[str] = set()
    for row in list(baseline_rows.values()) + list(candidate_rows.values()):
        names.update(row.metrics.keys())
    return names


def _compare_metric(
    metric: str,
    baseline_rows: Mapping[str, RetrievalRow],
    candidate_rows: Mapping[str, RetrievalRow],
    shared_case_ids: Sequence[str],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> MetricComparison:
    pairs = _paired_values(metric, baseline_rows, candidate_rows, shared_case_ids)
    if metric == "latency_ms_warm" and len(pairs) > 1:
        pairs = pairs[1:]

    if not pairs:
        return MetricComparison(
            metric=metric,
            baseline_mean=None,
            candidate_mean=None,
            paired_delta=None,
            relative_delta=None,
            wins=0,
            ties=0,
            losses=0,
            ci_low=None,
            ci_high=None,
            shared_count=0,
        )

    baseline_values = [baseline for baseline, _candidate in pairs]
    candidate_values = [candidate for _baseline, candidate in pairs]
    deltas = [candidate - baseline for baseline, candidate in pairs]
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    paired_delta = statistics.fmean(deltas)
    relative_delta = (
        paired_delta / baseline_mean if abs(baseline_mean) > FLOAT_TOLERANCE else None
    )
    wins, ties, losses = _win_tie_loss(metric, pairs)
    ci_low, ci_high = _bootstrap_mean_ci(
        deltas,
        samples=bootstrap_samples,
        seed=random_seed,
    )

    return MetricComparison(
        metric=metric,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        paired_delta=paired_delta,
        relative_delta=relative_delta,
        wins=wins,
        ties=ties,
        losses=losses,
        ci_low=ci_low,
        ci_high=ci_high,
        shared_count=len(pairs),
    )


def _paired_values(
    metric: str,
    baseline_rows: Mapping[str, RetrievalRow],
    candidate_rows: Mapping[str, RetrievalRow],
    shared_case_ids: Sequence[str],
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for case_id in shared_case_ids:
        baseline_value = _metric_value(baseline_rows[case_id], metric)
        candidate_value = _metric_value(candidate_rows[case_id], metric)
        if baseline_value is None or candidate_value is None:
            continue
        pairs.append((baseline_value, candidate_value))
    return pairs


def _win_tie_loss(metric: str, pairs: Sequence[tuple[float, float]]) -> tuple[int, int, int]:
    wins = ties = losses = 0
    lower_is_better = metric in LOWER_IS_BETTER
    for baseline, candidate in pairs:
        diff = candidate - baseline
        if abs(diff) <= FLOAT_TOLERANCE:
            ties += 1
        elif (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better):
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        value = float(values[0])
        return value, value

    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return means[lower_index], means[upper_index]


def _print_missing_cases(title: str, cases: Sequence[str]) -> None:
    if not cases:
        return
    print(f"\n{title} ({len(cases)}):")
    for case_id in cases[:20]:
        print(f"  - {case_id}")
    if len(cases) > 20:
        print(f"  ... {len(cases) - 20} more")


def _print_metric_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        print("No metrics found.")
        return

    print(
        "metric | baseline | candidate | paired_delta | relative_delta | "
        "wins/ties/losses | 95% CI | n"
    )
    print(
        "-------|----------|-----------|--------------|----------------|"
        "------------------|--------|---"
    )
    for row in rows:
        print(
            f"{row['metric']} | "
            f"{_fmt_number(row['baseline_mean'])} | "
            f"{_fmt_number(row['candidate_mean'])} | "
            f"{_fmt_number(row['paired_delta'], signed=True)} | "
            f"{_fmt_percent(row['relative_delta'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"[{_fmt_number(row['ci_low'], signed=True)}, "
            f"{_fmt_number(row['ci_high'], signed=True)}] | "
            f"{row['shared_count']}"
        )


def _fmt_number(value: Any, *, signed: bool = False) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


def _fmt_percent(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.2%}"


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retrieval evaluation results by shared case_id."
    )
    parser.add_argument("baseline", help="Baseline retrieval-result JSON")
    parser.add_argument("candidate", help="Candidate retrieval-result JSON")
    parser.add_argument("--baseline-mode", help="Expected mode in baseline rows")
    parser.add_argument("--candidate-mode", help="Expected mode in candidate rows")
    parser.add_argument("--baseline-method", help="Expected retrieval_method in baseline packets")
    parser.add_argument("--candidate-method", help="Expected retrieval_method in candidate packets")
    parser.add_argument(
        "--strict-case-ids",
        action="store_true",
        help="Fail if the two files do not contain identical case_id sets.",
    )
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Include config-like fields such as top_k and retrieved_count.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2_000,
        help="Number of bootstrap samples for paired mean-delta CI.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Bootstrap random seed")
    parser.add_argument("--json-output", help="Optional path for comparison JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    baseline_payload = load_payload(args.baseline)
    candidate_payload = load_payload(args.candidate)

    validate_payload_contract(
        baseline_payload,
        expected_mode=args.baseline_mode,
        expected_method=args.baseline_method,
        label="baseline",
    )
    validate_payload_contract(
        candidate_payload,
        expected_mode=args.candidate_mode,
        expected_method=args.candidate_method,
        label="candidate",
    )

    comparison = compare_payloads(
        baseline_payload,
        candidate_payload,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
        include_config=args.include_config,
    )

    if args.strict_case_ids and (
        comparison["baseline_only_cases"] or comparison["candidate_only_cases"]
    ):
        raise SystemExit(
            "Case IDs differ. Use the same fixture for method comparisons. "
            f"Baseline-only={comparison['baseline_only_cases']}; "
            f"candidate-only={comparison['candidate_only_cases']}"
        )

    print_comparison_report(
        comparison,
        baseline_label=args.baseline,
        candidate_label=args.candidate,
    )

    if args.json_output:
        _write_json(args.json_output, comparison)


if __name__ == "__main__":
    main()
