from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import create_agent
from app.agents.state import create_initial_state
from app.config import settings

RESULTS_DIR = Path("evaluation/results")

NODE_ORDER = [
    "research_news",
    "fetch_stock",
    "retrieve_filings",
    "analyze_sentiment",
    "draft_memo",
    "verify_memo",
]

RETRIEVAL_PHASE = {"research_news", "fetch_stock", "retrieve_filings", "analyze_sentiment"}
SERIAL_TAIL = {"draft_memo", "verify_memo"}

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    s = sorted(values)
    if len(s) == 1:
        return s[0]

    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"

async def time_single_run(
    agent,
    ticker: str,
    *,
    include_filing_analysis: bool,
    include_news_sentiment: bool,
    max_news_articles: int,
) -> dict:
    """Run once, attributing wall-clock between stream updates to each node.

    stream_mode="updates" yields {node_name: state_update} as each node
    completes; the elapsed time since the previous update approximates that
    node's duration. No tracer is attached, to keep the measurement clean.
    """
    state = create_initial_state(
        ticker,
        None,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )

    node_durations: dict[str, float] = {}
    t_start = time.perf_counter()
    t_prev = t_start

    async for update in agent.astream(state, stream_mode="updates", config={}):
        now = time.perf_counter()
        delta_ms = (now - t_prev) * 1000.0
        for node_name in update.keys():
            node_durations[node_name] = node_durations.get(node_name, 0.0) + delta_ms

        t_prev = now

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return {"total_ms": total_ms, "nodes": node_durations}

def _projected_parallel_total(node_p50: dict[str, float]) -> dict[str, float]:
    """Dependency-respecting best-case parallel projection from warm p50s.

    Shape modeled:
        fetch_stock                         (first, resolves company_name)
        then in parallel:
            branch A: research_news -> analyze_sentiment
            branch B: retrieve_filings
        then: draft_memo -> verify_memo  (serial tail, untouched)
    """

    fetch = node_p50.get("fetch_stock", 0.0)
    branch_a = node_p50.get("research_news", 0.0) + node_p50.get("analyze_sentiment", 0.0)
    branch_b = node_p50.get("retrieve_filings", 0.0)
    parallel_phase = fetch + max(branch_a, branch_b)
    tail = sum(node_p50.get(n, 0.0) for n in SERIAL_TAIL)
    return {
        "projected_retrieval_phase_ms": parallel_phase,
        "serial_tail_ms": tail,
        "projected_total_ms": parallel_phase + tail,
    }

async def main() -> None:
    parser = argparse.ArgumentParser(description="Serial latency baseline")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--repeats", type=int, default=5, help="runs per ticker (run 1 = cold)")
    parser.add_argument("--no-filings", action="store_true")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--max-news", type=int, default=10)
    args = parser.parse_args()

    include_filing_analysis = not args.no_filings
    include_news_sentiment = not args.no_sentiment

    agent = create_agent()

    raw_runs: list[dict] = []
    for ticker in args.tickers:
        for i in range(args.repeats):
            result = await time_single_run(
                agent,
                ticker,
                include_filing_analysis=include_filing_analysis,
                include_news_sentiment=include_news_sentiment,
                max_news_articles=args.max_news,
            )
            result["ticker"] = ticker
            result["run_index"] = i
            result["cold"] = i == 0
            raw_runs.append(result)
            tag = "cold" if i == 0 else "warm"
            print(f"  {ticker} run {i} ({tag}): {result['total_ms']:.0f} ms")

    warm = [r for r in raw_runs if not r["cold"]]
    if not warm:
        print("No warm runs (need --repeats >= 2). Aborting aggregation.")
        return

    node_stats: dict[str, dict[str, float]] = {}
    node_p50: dict[str, float] = {}
    for node in NODE_ORDER:
        vals = [r["nodes"].get(node, 0.0) for r in warm if node in r["nodes"]]
        if not vals:
            continue
        node_stats[node] = {
            "mean_ms": sum(vals) / len(vals),
            "p50_ms": _percentile(vals, 0.50),
            "p95_ms": _percentile(vals, 0.95),
            "n": len(vals),
        }
        node_p50[node] = node_stats[node]["p50_ms"]

    totals_warm = [r["total_ms"] for r in warm]
    totals_cold = [r["total_ms"] for r in raw_runs if r["cold"]]

    serial_retrieval_phase = sum(node_p50.get(n, 0.0) for n in RETRIEVAL_PHASE)
    serial_tail = sum(node_p50.get(n, 0.0) for n in SERIAL_TAIL)
    projection = _projected_parallel_total(node_p50)
    serial_total_p50 = _percentile(totals_warm, 0.50)
    projected_savings = serial_total_p50 - projection["projected_total_ms"]
    projected_pct = (projected_savings / serial_total_p50 * 100.0) if serial_total_p50 else 0.0

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provider": settings.llm.provider,
        "model": settings.llm.model if settings.llm.provider == "bedrock" else settings.ollama.llm_model,
        "config": {
            "tickers": args.tickers,
            "repeats": args.repeats,
            "include_filing_analysis": include_filing_analysis,
            "include_news_sentiment": include_news_sentiment,
            "max_news_articles": args.max_news,
        },
        "warm": {
            "total_p50_ms": serial_total_p50,
            "total_p95_ms": _percentile(totals_warm, 0.95),
            "n_runs": len(warm),
            "per_node": node_stats,
            "serial_retrieval_phase_p50_ms": serial_retrieval_phase,
            "serial_tail_p50_ms": serial_tail,
        },
        "cold": {
            "total_values_ms": totals_cold,
        },
        "parallel_projection": {
            **projection,
            "projected_savings_ms": projected_savings,
            "projected_improvement_pct": projected_pct,
        },
        "raw_runs": raw_runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"latency_baseline_{stamp}.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n=== SERIAL LATENCY BASELINE (warm) ===")
    print(f"provider/model : {artifact['provider']} / {artifact['model']}")
    print(f"warm total     : p50 {serial_total_p50:.0f} ms | p95 {artifact['warm']['total_p95_ms']:.0f} ms (n={len(warm)})")
    print(f"cold total     : {', '.join(f'{v:.0f}' for v in totals_cold)} ms")
    print("\nper-node (warm p50 / p95 ms):")
    for node in NODE_ORDER:
        if node in node_stats:
            s = node_stats[node]
            phase = "retrieval" if node in RETRIEVAL_PHASE else "tail"
            print(f"  {node:<20} {s['p50_ms']:>8.0f} / {s['p95_ms']:>8.0f}   [{phase}]")
    print(f"\nserial retrieval phase (parallelizable) : {serial_retrieval_phase:.0f} ms")
    print(f"serial LLM tail (untouchable)           : {serial_tail:.0f} ms")
    print("\n--- parallelism ceiling (projected, not yet built) ---")
    print(f"projected total with parallelism : {projection['projected_total_ms']:.0f} ms")
    print(f"projected savings                : {projected_savings:.0f} ms ({projected_pct:.1f}%)")

    print(f"\nartifact: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
