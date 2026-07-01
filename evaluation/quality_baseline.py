"""Single-graph quality baseline harness (semantic-oracle edition).

Scores memos with the calibrated semantic grounding checker instead of reading the
runtime (token-overlap) verification_result. Records the checker + threshold into the
artifact, because these numbers are NOT comparable to earlier token-checker baselines.

NOTE: the graph's verify_memo node still runs the token checker at runtime, so
error_steps={"verify_memo": N} reflects the OLD checker's failures, not the semantic
verdict. Ignore it when reading the semantic metrics; it's runtime provenance only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import create_agent
from app.agents.state import create_initial_state, has_fatal_error
from app.config import settings
from app.llm.provider import MODEL_PRESETS, _model_family
from evaluation.grounding import evaluate_memo_grounding
from evaluation.semantic_grounding import evaluate_memo_grounding_semantic

RESULTS_DIR = Path("evaluation/results")
MIN_CITATION_COVERAGE = 0.80
MIN_GROUNDED_CLAIM_RATE = 0.75
_CIT_RE = re.compile(r"\[(\d+)\]")


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


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": _percentile(values, 0.50),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _git_state() -> dict:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""
    return {"commit": run(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
            "dirty": bool(run(["git", "status", "--porcelain"]))}


async def evaluate_single_run(agent, ticker, *, include_filing_analysis, include_news_sentiment,
                              max_news_articles, checker: str, min_similarity: float) -> dict:
    state = create_initial_state(
        ticker, None,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )
    t0 = time.perf_counter()
    final = await agent.ainvoke(state, config={})
    run_ms = (time.perf_counter() - t0) * 1000.0

    memo = final.get("investment_memo", "") or ""
    registry = final.get("citation_evidence", []) or []
    errors = final.get("errors", []) or []

    if checker == "semantic":
        gr = evaluate_memo_grounding_semantic(memo, registry, min_similarity=min_similarity).to_dict()
    else:
        gr = evaluate_memo_grounding(memo, registry).to_dict()

    used = {int(m) for m in _CIT_RE.findall(memo)}
    valid = {e.get("index") for e in registry if isinstance(e, dict)}
    orphans = len(used - valid)

    return {
        "ticker": ticker,
        "passed": bool(gr["passed"]),
        "total_claims": int(gr["total_claims"]),
        "cited_claims": int(gr["cited_claims"]),
        "grounded_claims": int(gr["grounded_claims"]),
        "citation_coverage_rate": float(gr["citation_coverage_rate"]),
        "grounded_claim_rate": float(gr["grounded_claim_rate"]),
        "orphan_citations": orphans,
        "evidence_count": len(registry),
        "filing_chunks": len(final.get("filing_chunks", []) or []),
        "news_articles": len(final.get("news_articles", []) or []),
        "memo_words": len(memo.split()),
        "n_errors": len(errors),
        "errors": [{"step": e.get("step"), "message": (e.get("message") or "")[:200]} for e in errors],
        "fatal_error": has_fatal_error(final),
        "run_ms": run_ms,
    }


def _aggregate(runs: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in runs]
    n = len(runs)
    keys = ["grounded_claim_rate", "citation_coverage_rate", "total_claims", "cited_claims",
            "grounded_claims", "orphan_citations", "evidence_count", "filing_chunks",
            "news_articles", "memo_words", "run_ms"]
    out = {"n_runs": n,
           "pass_rate": sum(1 for r in runs if r["passed"]) / n if n else 0.0,
           "fatal_error_rate": sum(1 for r in runs if r["fatal_error"]) / n if n else 0.0}
    for k in keys:
        out[k] = _stats(col(k))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description="Single-graph quality baseline")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checker", choices=["semantic", "token"], default="semantic")
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--no-filings", action="store_true")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--max-news", type=int, default=10)
    args = parser.parse_args()

    include_filing_analysis = not args.no_filings
    include_news_sentiment = not args.no_sentiment
    agent = create_agent()

    runs: list[dict] = []
    for ticker in args.tickers:
        for i in range(args.repeats):
            r = await evaluate_single_run(
                agent, ticker,
                include_filing_analysis=include_filing_analysis,
                include_news_sentiment=include_news_sentiment,
                max_news_articles=args.max_news,
                checker=args.checker, min_similarity=args.min_similarity,
            )
            runs.append(r)
            print(f"  {ticker} run {i}: grounded={r['grounded_claim_rate']:.2f} "
                  f"coverage={r['citation_coverage_rate']:.2f} claims={r['total_claims']} passed={r['passed']}")

    overall = _aggregate(runs)
    error_steps = Counter(e["step"] for r in runs for e in r["errors"])
    model_id = MODEL_PRESETS.get(settings.llm.model, settings.llm.model)

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "provider": settings.llm.provider,
        "model": model_id,
        "model_family": _model_family(model_id) if settings.llm.provider == "bedrock" else settings.llm.provider,
        "temperature": settings.llm.temperature,
        "thinking_mode": settings.llm.thinking_mode,
        "grounding_checker": args.checker,
        "min_similarity": args.min_similarity if args.checker == "semantic" else None,
        "thresholds": {"min_citation_coverage": MIN_CITATION_COVERAGE, "min_grounded_claim_rate": MIN_GROUNDED_CLAIM_RATE},
        "config": {"tickers": args.tickers, "repeats": args.repeats,
                   "include_filing_analysis": include_filing_analysis,
                   "include_news_sentiment": include_news_sentiment, "max_news_articles": args.max_news},
        "overall": overall,
        "error_steps": dict(error_steps),
        "per_ticker": {t: _aggregate([r for r in runs if r["ticker"] == t]) for t in args.tickers},
        "raw_runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"quality_baseline_{stamp}.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    g, c = overall["grounded_claim_rate"], overall["citation_coverage_rate"]
    dirty = " (WORKING TREE DIRTY)" if artifact["git"]["dirty"] else ""
    print("\n=== QUALITY BASELINE ===")
    print(f"model/family/temp   : {model_id} / {artifact['model_family']} / {artifact['temperature']}")
    print(f"grounding_checker   : {args.checker}" + (f" (min_similarity={args.min_similarity})" if args.checker == 'semantic' else ""))
    print(f"thinking_mode       : {artifact['thinking_mode']}   commit {artifact['git']['commit']}{dirty}")
    print(f"runs                : {overall['n_runs']}  (pass_rate {overall['pass_rate']:.2f})")
    print(f"grounded_claim_rate : {g['mean']:.3f} +/- {g['stdev']:.3f}")
    print(f"citation_coverage   : {c['mean']:.3f} +/- {c['stdev']:.3f}")
    print(f"runtime error steps : {dict(error_steps) or 'none'}  (token-checker provenance)")
    print(f"\nartifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())