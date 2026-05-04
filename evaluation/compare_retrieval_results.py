"""Compare retrieval benchmark result JSON files.

Usage:
    uv run python evaluation/compare_retrieval_results.py \
        evaluation/results/chroma_relevance.json \
        evaluation/results/qdrant_relevance.json

Prints absolute and relative deltas for shared numeric metrics.
Relative delta is:
    ((candidate - baseline) / baseline) * 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_METRICS = (
    "precision_at_5",
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "section_hit_at_5",
    "citation_support_rate",
    "average_latency_ms",
    "latency_ms",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(payload).__name__}")

    return payload


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_metric_name(name: str) -> str:
    return (
        name.lower()
        .replace("@", "_at_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def flatten_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for key, value in payload.items():
        normalized_key = normalize_metric_name(key)
        if is_number(value):
            metrics[normalized_key] = float(value)

    nested = payload.get("metrics")
    if isinstance(nested, dict):
        for key, value in nested.items():
            normalized_key = normalize_metric_name(key)
            if is_number(value):
                metrics[normalized_key] = float(value)

    return metrics


def extract_per_query_metrics(payload: dict[str, Any]) -> dict[str, float]:
    rows: Any = None

    for key in ("results", "queries", "records", "items"):
        if isinstance(payload.get(key), list):
            rows = payload[key]
            break

    if not rows:
        return {}

    values_by_metric: dict[str, list[float]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        row_metrics = flatten_numeric_metrics(row)

        nested = row.get("metrics")
        if isinstance(nested, dict):
            row_metrics.update(flatten_numeric_metrics({"metrics": nested}))

        for key, value in row_metrics.items():
            values_by_metric.setdefault(key, []).append(value)

    return {
        key: mean(values)
        for key, values in values_by_metric.items()
        if values
    }


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = flatten_numeric_metrics(payload)

    # If aggregate metrics are not present, derive averages from per-query rows.
    per_query_metrics = extract_per_query_metrics(payload)
    for key, value in per_query_metrics.items():
        metrics.setdefault(key, value)

    return metrics


def relative_delta_pct(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def compare_metrics(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    metric_filter: list[str] | None = None,
) -> list[tuple[str, float, float, float, float | None]]:
    if metric_filter:
        metric_names = [normalize_metric_name(metric) for metric in metric_filter]
    else:
        preferred = [normalize_metric_name(metric) for metric in DEFAULT_METRICS]
        available = sorted(set(baseline_metrics) | set(candidate_metrics))
        metric_names = [metric for metric in preferred if metric in available]
        metric_names.extend(metric for metric in available if metric not in metric_names)

    rows: list[tuple[str, float, float, float, float | None]] = []

    for metric in metric_names:
        if metric not in baseline_metrics or metric not in candidate_metrics:
            continue

        baseline = baseline_metrics[metric]
        candidate = candidate_metrics[metric]
        absolute_delta = candidate - baseline
        relative_delta = relative_delta_pct(baseline, candidate)

        rows.append((metric, baseline, candidate, absolute_delta, relative_delta))

    return rows


def print_table(rows: list[tuple[str, float, float, float, float | None]]) -> None:
    if not rows:
        print("No shared numeric metrics found to compare.")
        return

    headers = ("metric", "baseline", "candidate", "absolute_delta", "relative_delta")
    table_rows = [
        (
            metric,
            format_float(baseline),
            format_float(candidate),
            format_float(absolute_delta),
            format_pct(relative_delta),
        )
        for metric, baseline, candidate, absolute_delta, relative_delta in rows
    ]

    widths = [
        max(len(str(row[i])) for row in (headers, *table_rows))
        for i in range(len(headers))
    ]

    def render_row(row: tuple[str, ...]) -> str:
        return " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))

    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(render_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two retrieval benchmark result files.")
    parser.add_argument("baseline", type=Path, help="Baseline result JSON, e.g. Chroma.")
    parser.add_argument("candidate", type=Path, help="Candidate result JSON, e.g. Qdrant.")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to compare. Can be passed multiple times. Defaults to all shared metrics.",
    )
    args = parser.parse_args()

    baseline_payload = load_json(args.baseline)
    candidate_payload = load_json(args.candidate)

    baseline_metrics = extract_metrics(baseline_payload)
    candidate_metrics = extract_metrics(candidate_payload)

    rows = compare_metrics(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        metric_filter=args.metrics,
    )

    print(f"Baseline:  {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print()
    print_table(rows)


if __name__ == "__main__":
    main()
