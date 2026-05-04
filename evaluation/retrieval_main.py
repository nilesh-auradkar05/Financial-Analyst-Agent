from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from evaluation.retrieval_eval import (
    RetrievalEvalCase,
    RetrievalEvalResult,
    evaluate_retrieval_cases,
    load_retrieval_cases,
    save_retrieval_results,
    summarize_retrieval_results,
)

SUPPORTED_MODES = {
    "query",
    "sections",
    "naive",
    "semantic",
    "section_aware",
    "section-aware",
    "hybrid",
    "langchain_hybrid",
    "qdrant_hybrid",
    "reranked_hybrid",
}


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("evaluation/reports") / f"retrieval_eval_{ts}.json"


def _normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "section-aware": "section_aware",
        "qdrant_hybrid": "hybrid",
        "langchain_hybrid": "hybrid",
        "query": "semantic",
    }
    return aliases.get(normalized, normalized)


def _apply_mode_override(
    cases: Iterable[RetrievalEvalCase],
    mode: str | None,
) -> list[RetrievalEvalCase]:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode is None:
        return list(cases)
    if normalized_mode not in {_normalize_mode(item) for item in SUPPORTED_MODES}:
        valid_modes = ", ".join(sorted({_normalize_mode(item) or item for item in SUPPORTED_MODES}))
        raise SystemExit(f"Unsupported retrieval mode: {mode}. Valid modes: {valid_modes}")
    return [replace(case, mode=normalized_mode) for case in cases]


def _print_result_row(result: RetrievalEvalResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    first_rank = result.metrics.first_relevant_rank
    first_rank_text = "MISS" if first_rank is None else str(first_rank)
    print(
        f"{status:4} | {result.case_id:36} | "
        f"mode={result.mode:16} | "
        f"latency={result.metrics.latency_ms:8.1f} | "
        f"p@5={result.metrics.precision_at_5:5.2f} | "
        f"r@5={result.metrics.recall_at_5:5.2f} | "
        f"mrr@5={result.metrics.mrr_at_5:5.2f} | "
        f"ndcg@5={result.metrics.ndcg_at_5:5.2f} | "
        f"section_recall={result.metrics.section_recall_at_k:5.2f} | "
        f"keyword_hit={result.metrics.keyword_hit_rate:5.2f} | "
        f"first_rank={first_rank_text}"
    )
    if result.issues:
        print(f"\tissues: {', '.join(result.issues)}")
    if result.error:
        print(f"\terror: {result.error}")


def _print_summary(summary: dict[str, object]) -> None:
    print("=" * 120)
    print(
        f"pass_rate={float(summary['pass_rate']):.2f} | "
        f"avg_latency_ms={float(summary['avg_latency_ms']):.1f} | "
        f"avg_p@5={float(summary.get('avg_precision_at_5', 0.0)):.2f} | "
        f"avg_r@5={float(summary.get('avg_recall_at_5', 0.0)):.2f} | "
        f"avg_mrr@5={float(summary.get('avg_mrr_at_5', 0.0)):.2f} | "
        f"avg_ndcg@5={float(summary.get('avg_ndcg_at_5', 0.0)):.2f} | "
        f"avg_section_recall@k={float(summary['avg_section_recall_at_k']):.2f} | "
        f"avg_keyword_hit_rate={float(summary['avg_keyword_hit_rate']):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation fixtures")
    parser.add_argument(
        "fixtures",
        help="Path to retrieval fixture JSON file",
    )
    parser.add_argument(
        "--case",
        help="Run only a single case by id",
    )
    parser.add_argument(
        "--mode",
        choices=sorted({_normalize_mode(item) or item for item in SUPPORTED_MODES}),
        help=(
            "Override fixture retrieval mode for every case. Use this for fair same-fixture "
            "method comparisons, e.g. semantic, section_aware, hybrid, reranked_hybrid."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="Path for JSON report output",
    )
    args = parser.parse_args()

    cases = load_retrieval_cases(args.fixtures)
    if args.case:
        cases = [case for case in cases if case.id == args.case]
        if not cases:
            raise SystemExit(f"Unknown case id: {args.case}")

    cases = _apply_mode_override(cases, args.mode)

    results = evaluate_retrieval_cases(cases)
    summary = summarize_retrieval_results(results)

    print("Retrieval Evaluation Results")
    print("=" * 120)
    for result in results:
        _print_result_row(result)
    _print_summary(summary)

    report_path = save_retrieval_results(results, args.output)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
