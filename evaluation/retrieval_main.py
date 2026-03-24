from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from evaluation.retrieval_eval import (
    evaluate_retrieval_cases,
    load_retrieval_cases,
    save_retrieval_results,
    summarize_retrieval_results,
)


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("evaluation/reports") / f"retrieval_eval_{ts}.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation fixtures")
    parser.add_argument(
        "fixtures",
        help="Path to retrieval fixture JSON File",
    )
    parser.add_argument(
        "--case",
        help="Run only a single case by id",
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

    results = evaluate_retrieval_cases(cases)
    summary = summarize_retrieval_results(results)

    print("Retrieval Evaluation Results")
    print("=" * 70)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:4} | {result.case_id:30} | "
            f"latency={result.metrics.latency_ms:7.1f} | "
            f"section_hit={str(result.metrics.section_hit_at_k):5} | "
            f"section_recall={result.metrics.section_recall_at_k:6.2f} | "
            f"keyword_hit_rate={result.metrics.keyword_hit_rate:.2f} | "
            f"first_rank={result.metrics.first_relevant_rank}"
        )
        if result.issues:
            print(f"\t\tissues: {', '.join(result.issues)}")

    print("=" * 70)
    print(
        f"pass_rate={summary['pass_rate']:.2f} | "
        f"avg_latency_ms={summary['avg_latency_ms']:.1f} | "
        f"avg_section_recall@k={summary['avg_section_recall_at_k']:.2f} | "
        f"avg_keyword_hit_rate={summary['avg_keyword_hit_rate']:.2f}"
    )

    report_path = save_retrieval_results(results, args.output)
    print(f"Saved report: {report_path}")

if __name__ == "__main__":
    main()
