"""Export one JSON line per live analysis run from LangSmith traces.

Reads the trace of each `run_financial_analysis` run and prints per-node timings,
ingestion, LLM usage, evidence counts, errors and verification, so runs from
different code versions can be compared or charted from one JSONL file.

The LangSmith key lives in the api container, so run it there and append on the host:

    docker compose exec -T api python - --label ingest-on-miss --since-hours 2 \
        < evaluation/langsmith_run_metrics.py >> evaluation/results/live_runs.jsonl

    # specific traces
    docker compose exec -T api python - --label baseline --trace <trace_id> \
        < evaluation/langsmith_run_metrics.py >> evaluation/results/live_runs.jsonl

These are live-evidence probes (n=1 per run, network and provider noise), not
frozen-release results; compare like with like (same ticker, same model).
Traces carry no git revision, so `--label` names the code version by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Iterable

SCHEMA_VERSION = 1
NODES = (
    "research_news",
    "fetch_stock",
    "retrieve_filings",
    "analyze_sentiment",
    "draft_memo",
    "verify_memo",
)
EVIDENCE_NODES = NODES[:3]
# Root-run durations below this are rejected requests or test traffic, not analyses.
MIN_ANALYSIS_SECONDS = 5.0


def _seconds(run: Any) -> float | None:
    return round((run.end_time - run.start_time).total_seconds(), 3) if run.end_time else None


def _first(runs: list[Any], name: str) -> Any | None:
    return next((run for run in runs if run.name == name), None)


def summarize(runs: Iterable[Any], label: str) -> dict[str, Any] | None:
    """Reduce one trace's runs to a flat record. None when the trace holds no analysis."""
    ordered = sorted(runs, key=lambda run: run.start_time)
    analysis = _first(ordered, "run_financial_analysis")
    if analysis is None:
        return None
    t0 = analysis.start_time
    inputs = analysis.inputs or {}
    outputs = analysis.outputs or {}
    metadata = (analysis.extra or {}).get("metadata", {})

    # Each node is traced twice (graph node and app span) with the same times; keep the first.
    nodes: dict[str, dict[str, float | None]] = {}
    for name in NODES:
        if run := _first(ordered, name):
            nodes[name] = {
                "start_s": round((run.start_time - t0).total_seconds(), 3),
                "duration_s": _seconds(run),
            }
    spans = [
        (run.start_time, run.end_time)
        for name in EVIDENCE_NODES
        if (run := _first(ordered, name)) and run.end_time
    ]
    evidence_phase = (
        round((max(end for _, end in spans) - min(start for start, _ in spans)).total_seconds(), 3)
        if spans
        else None
    )

    ingest = _first(ordered, "ingest_10k_for_ticker")
    ingest_meta = (ingest.extra or {}).get("metadata", {}) if ingest else {}
    llm = next((run for run in ordered if run.run_type == "llm"), None)
    attempts = [
        {"name": run.name, "status": run.status, "duration_s": _seconds(run)}
        for run in ordered
        if run.name.endswith("_attempt")
    ]
    verification = outputs.get("verification_result") or {}
    filing_chunks = outputs.get("filing_chunks") or []

    return {
        "schema": SCHEMA_VERSION,
        "label": label,
        "trace_id": str(analysis.trace_id),
        "started_at": analysis.start_time.isoformat(),
        "ticker": outputs.get("ticker") or inputs.get("ticker"),
        "company_name": outputs.get("company_name") or None,
        "include_filing_analysis": inputs.get("include_filing_analysis"),
        "include_news_sentiment": inputs.get("include_news_sentiment"),
        "max_news_articles": inputs.get("max_news_articles"),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "trace_status": analysis.status,
        "total_s": _seconds(analysis),
        "evidence_phase_s": evidence_phase,
        "nodes": nodes,
        "filing_searches": sum(run.name == "search_filing_chunks" for run in ordered),
        "ingest": {
            "ran": ingest is not None,
            "duration_s": _seconds(ingest) if ingest else None,
            "fetch_10k_s": _seconds(run) if (run := _first(ordered, "get_latest_10k")) else None,
            "embed_and_write_s": _seconds(run)
            if (run := _first(ordered, "ingest_filing"))
            else None,
            "outcome": ingest_meta.get("outcome"),
            "error_code": ingest_meta.get("error_code"),
        },
        "llm": {
            "model": ((llm.extra or {}).get("metadata", {}).get("ls_model_name") if llm else None),
            "duration_s": _seconds(llm) if llm else None,
            "input_tokens": llm.prompt_tokens if llm else None,
            "output_tokens": llm.completion_tokens if llm else None,
            "total_tokens": llm.total_tokens if llm else None,
            "cost_usd": float(llm.total_cost) if llm and llm.total_cost is not None else None,
            "attempts": attempts,
        },
        "evidence": {
            "news_articles": len(outputs.get("news_articles") or []),
            "filing_chunks": len(filing_chunks),
            "filing_sections": sorted({chunk.get("section", "Unknown") for chunk in filing_chunks}),
            "has_stock_data": bool(outputs.get("stock_data")),
            "sentiment": (outputs.get("sentiment_result") or {}).get("overall_sentiment"),
            "citations": len(outputs.get("citations") or []),
        },
        "errors": [
            {
                "step": error.get("step"),
                "code": error.get("code"),
                "recoverable": error.get("recoverable"),
            }
            for error in outputs.get("errors") or []
        ],
        "verification": {
            key: verification.get(key)
            for key in (
                "passed",
                "total_claims",
                "cited_claims",
                "grounded_claims",
                "citation_coverage_rate",
                "grounded_claim_rate",
                "orphan_citations",
            )
        },
        "memo_words": len((outputs.get("investment_memo") or "").split()),
    }


def _self_check() -> None:
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    def span(
        name: str, start: float, duration: float, run_type: str = "chain", **extra: Any
    ) -> SimpleNamespace:
        defaults: dict[str, Any] = dict(
            name=name,
            run_type=run_type,
            start_time=t0 + dt.timedelta(seconds=start),
            end_time=t0 + dt.timedelta(seconds=start + duration),
            inputs={},
            outputs={},
            extra={},
            status="success",
            trace_id="t",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            total_cost=None,
        )
        return SimpleNamespace(**{**defaults, **extra})

    record = summarize(
        [
            span(
                "run_financial_analysis",
                0,
                30,
                inputs={"ticker": "AAPL"},
                outputs={
                    "ticker": "AAPL",
                    "filing_chunks": [{"section": "Risk"}],
                    "errors": [{"step": "verify_memo"}],
                },
            ),
            span("research_news", 0, 5),
            span("fetch_stock", 0, 1),
            span("retrieve_filings", 0, 8),
            span("retrieve_filings", 0, 8),
            span("ingest_10k_for_ticker", 1, 6),
            span("bedrock_attempt", 10, 20),
            span("Chat", 10, 20, run_type="llm", total_tokens=7, total_cost=0.5),
        ],
        "check",
    )
    assert record is not None
    assert record["evidence_phase_s"] == 8.0, record[
        "evidence_phase_s"
    ]  # max of parallel branches, not the sum
    assert record["ingest"] == {**record["ingest"], "ran": True, "duration_s": 6.0}
    assert (
        record["llm"]["total_tokens"] == 7
        and record["llm"]["attempts"][0]["name"] == "bedrock_attempt"
    )
    assert record["evidence"]["filing_chunks"] == 1 and record["errors"][0]["step"] == "verify_memo"
    assert summarize([span("HTTP GET /health", 0, 0.1)], "check") is None
    print("self-check ok", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", help="name of the code version these runs came from")
    parser.add_argument("--trace", action="append", default=[], help="trace id (repeatable)")
    parser.add_argument(
        "--since-hours", type=float, help="export every analysis started in the last N hours"
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return _self_check()
    if not args.label or not (args.trace or args.since_hours):
        parser.error("--label and one of --trace / --since-hours are required")

    from langsmith import Client

    client = Client()
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    trace_ids = list(args.trace)
    if args.since_hours:
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.since_hours)
        found = client.list_runs(
            project_name=project, start_time=since, filter='eq(name, "run_financial_analysis")'
        )
        trace_ids += [
            str(run.trace_id) for run in found if (_seconds(run) or 0) >= MIN_ANALYSIS_SECONDS
        ]

    for trace_id in sorted(set(trace_ids)):  # trace ids are time-ordered
        record = summarize(client.list_runs(project_name=project, trace_id=trace_id), args.label)
        if record is None:
            print(f"no analysis run in trace {trace_id}", file=sys.stderr)
            continue
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
