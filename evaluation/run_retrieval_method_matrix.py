"""Run one retrieval fixture across multiple retrieval modes and compare them.

This is the safe way to answer questions like "does reranked hybrid beat
section-aware retrieval?" because every method is run on the exact same fixture
rows and therefore the result files share case IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from evaluation.compare_retrieval_results import compare_payloads, load_payload
from evaluation.retrieval_eval import (
    evaluate_retrieval_cases,
    load_retrieval_cases,
    save_retrieval_results,
)
from rag.vector_store import get_vector_store

DEFAULT_MODES = [
    "naive",
    "semantic",
    "section_aware",
    "hybrid",
    "reranked_hybrid",
]

MODE_ALIASES = {
    "dense": "semantic",
    "query": "semantic",
    "section-aware": "section_aware",
    "langchain_hybrid": "hybrid",
    "qdrant_hybrid": "hybrid",
    "hybrid_rerank": "reranked_hybrid",
    "dense_bm25_rerank": "reranked_hybrid",
}


def normalize_mode(mode: str) -> str:
    """Normalize user-facing mode aliases."""

    lowered = mode.strip().lower()
    return MODE_ALIASES.get(lowered, lowered)


def run_mode(
    *,
    fixture_path: Path,
    mode: str,
    output_path: Path,
    store: Any | None,
    case_id: str | None = None,
) -> Path:
    """Run one fixture with all cases forced to a single retrieval mode."""

    cases = load_retrieval_cases(fixture_path)
    if case_id:
        cases = [case for case in cases if case.id == case_id]
        if not cases:
            raise ValueError(f"Unknown case id: {case_id}")

    normalized_mode = normalize_mode(mode)
    overridden_cases = [replace(case, mode=normalized_mode) for case in cases]
    results = evaluate_retrieval_cases(overridden_cases, store=store)
    return save_retrieval_results(results, output_path)


def build_matrix_summary(
    *,
    result_paths: dict[str, Path],
    baseline_mode: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare every candidate mode against the selected baseline mode."""

    if baseline_mode not in result_paths:
        raise ValueError(
            f"Baseline mode {baseline_mode!r} was not run. Available: {sorted(result_paths)}"
        )

    baseline_payload = load_payload(result_paths[baseline_mode])
    comparisons: dict[str, Any] = {}
    for mode, path in result_paths.items():
        if mode == baseline_mode:
            continue
        candidate_payload = load_payload(path)
        comparisons[mode] = compare_payloads(
            baseline_payload,
            candidate_payload,
            bootstrap_samples=bootstrap_samples,
            random_seed=seed,
        )

    return {
        "baseline_mode": baseline_mode,
        "result_paths": {mode: str(path) for mode, path in result_paths.items()},
        "comparisons": comparisons,
    }


def print_matrix_summary(summary: dict[str, Any]) -> None:
    """Print compact matrix summary focused on key retrieval metrics."""

    baseline_mode = summary["baseline_mode"]
    print(f"Baseline mode: {baseline_mode}")
    print()
    print(
        "candidate | shared | recall@5 delta | ndcg@5 delta | "
        "section_recall delta | latency delta"
    )
    print(
        "----------|--------|----------------|--------------|"
        "----------------------|--------------"
    )
    for mode, comparison in summary["comparisons"].items():
        quality = {row["metric"]: row for row in comparison.get("quality", [])}
        runtime = {row["metric"]: row for row in comparison.get("runtime", [])}
        print(
            f"{mode} | "
            f"{comparison['shared_case_count']} | "
            f"{_fmt_delta(quality.get('recall_at_5'))} | "
            f"{_fmt_delta(quality.get('ndcg_at_5'))} | "
            f"{_fmt_delta(quality.get('section_recall_at_k'))} | "
            f"{_fmt_delta(runtime.get('latency_ms'))}"
        )


def _fmt_delta(row: dict[str, Any] | None) -> str:
    if not row or row.get("paired_delta") is None:
        return "n/a"
    relative = row.get("relative_delta")
    if relative is None:
        return f"{row['paired_delta']:+.4f}"
    return f"{row['paired_delta']:+.4f} ({relative:+.2%})"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one retrieval fixture across multiple retrieval modes."
    )
    parser.add_argument("fixture", help="Shared retrieval fixture JSON")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help="Retrieval modes to run. Defaults to common modes.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="section_aware",
        help="Mode used as baseline for pairwise comparisons.",
    )
    parser.add_argument("--case", help="Run only one case id")
    parser.add_argument(
        "--results-dir",
        default="evaluation/results/method_matrix",
        help="Directory where per-mode JSON outputs are saved.",
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix. Defaults to fixture stem plus vector backend.",
    )
    parser.add_argument(
        "--no-store-reuse",
        action="store_true",
        help="Create a fresh vector-store instance per mode.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--summary-output",
        help="Optional path for matrix comparison summary JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    vector_backend = os.getenv("VECTOR_BACKEND", "default")
    prefix = args.prefix or f"{fixture_path.stem}_{vector_backend}"
    modes = [normalize_mode(mode) for mode in args.modes]
    baseline_mode = normalize_mode(args.baseline_mode)

    shared_store = None if args.no_store_reuse else get_vector_store()
    result_paths: dict[str, Path] = {}

    for mode in modes:
        output_path = results_dir / f"{prefix}_{mode}.json"
        store = None if args.no_store_reuse else shared_store
        print(f"Running mode={mode} -> {output_path}")
        result_paths[mode] = run_mode(
            fixture_path=fixture_path,
            mode=mode,
            output_path=output_path,
            store=store,
            case_id=args.case,
        )

    summary = build_matrix_summary(
        result_paths=result_paths,
        baseline_mode=baseline_mode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print()
    print_matrix_summary(summary)

    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved matrix summary: {output_path}")


if __name__ == "__main__":
    main()
