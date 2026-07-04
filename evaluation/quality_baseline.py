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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import create_agent, draft_memo_node, verify_memo_node
from app.agents.state import AgentState, create_initial_state, has_fatal_error
from app.config import settings
from app.llm.provider import MODEL_PRESETS, _model_family
from dataops.evidence_release import SnapshotRecord
from dataops.git_state import git_state as _git_state
from dataops.replay import EvidenceReplayBundle, load_evidence_release_records
from evaluation.grounding import GROUNDING_EVAL_VERSION, evaluate_memo_grounding
from evaluation.semantic_grounding import evaluate_memo_grounding_semantic

RESULTS_DIR = Path("evaluation/results")
MIN_CITATION_COVERAGE = 0.80
MIN_GROUNDED_CLAIM_RATE = 0.75
_CIT_RE = re.compile(r"\[(\d+)\]")
_THRESHOLD_SWEEP = (0.40, 0.45, 0.50, 0.55)
_BORDERLINE_BAND = 0.05
_BASELINE_FILENAME_RE = re.compile(r"^retrieval_baseline_(\d{3})_")
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _cited_claims(claims: list[dict]) -> list[dict]:
    return [c for c in claims if not c.get("missing_citation")]


def _threshold_sensitivity(claims: list[dict], thresholds: tuple[float, ...] = _THRESHOLD_SWEEP) -> dict[str, float]:
    """Grounded rate at each similarity threshold, computed post-hoc from a single
    run's per-claim `overlap_score` (best cited similarity) + `numbers_ok` (union
    number gate). One run now answers the threshold-fragility question instead of
    needing repeated runs to sample noise around the 0.45 default.

    Meaningful for the semantic checker (overlap_score is cosine similarity); for
    the token checker overlap_score is a token-overlap ratio on the same 0-1 scale,
    so the same sweep is computed rather than special-cased to null.
    """
    cited = _cited_claims(claims)
    if not cited:
        return {f"{t:.2f}": 0.0 for t in thresholds}
    return {
        f"{t:.2f}": sum(1 for c in cited if c["overlap_score"] >= t and c["numbers_ok"]) / len(cited)
        for t in thresholds
    }


def _borderline_claims(claims: list[dict], min_similarity: float, band: float = _BORDERLINE_BAND) -> int:
    """Count of cited claims within `band` of the active threshold -- claims whose
    grounded/ungrounded verdict would flip under a small recalibration."""
    cited = _cited_claims(claims)
    return sum(1 for c in cited if abs(c["overlap_score"] - min_similarity) <= band)


def _sanitize_for_filename(value: str) -> str:
    return _FILENAME_SAFE_RE.sub("-", value)


def next_baseline_filename(results_dir: Path, model: str, temperature: float) -> str:
    """Pure helper for the next `retrieval_baseline_<NNN>_<model>_<temperature>.json`
    artifact name. NNN is a zero-padded 3-digit sequence = 1 + the highest existing
    NNN among `retrieval_baseline_*.json` files in `results_dir` (001 if none).
    `model` is sanitized to filesystem-safe characters.
    """
    seq = 0
    if results_dir.exists():
        for path in results_dir.glob("retrieval_baseline_*.json"):
            match = _BASELINE_FILENAME_RE.match(path.name)
            if match:
                seq = max(seq, int(match.group(1)))
    model_part = _sanitize_for_filename(model)
    return f"retrieval_baseline_{seq + 1:03d}_{model_part}_{float(temperature)}.json"


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


def _company_name_from_records(ticker: str, records: list[SnapshotRecord]) -> str | None:
    for record in records:
        value = record.payload.get("company_name")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _news_articles_from_records(records: list[SnapshotRecord], max_news_articles: int) -> list[dict]:
    articles: list[dict] = []
    for record in records:
        if record.snapshot.source_type != "news_article":
            continue
        payload = record.payload
        articles.append(
            {
                "title": str(payload.get("title") or "Untitled"),
                "url": str(payload.get("url") or record.snapshot.natural_key),
                "source": str(payload.get("source") or payload.get("publisher") or "Unknown"),
                "snippet": str(payload.get("snippet") or payload.get("content") or ""),
                "published_date": payload.get("published_date"),
            }
        )
    return articles[:max_news_articles]


def _stock_data_from_records(records: list[SnapshotRecord]) -> dict:
    market_records = [
        record for record in records if record.snapshot.source_type == "market_quote"
    ]
    if not market_records:
        return {}
    latest = max(market_records, key=lambda record: record.snapshot.fetched_at)
    return dict(latest.payload)


def _filing_chunks_from_records(records: list[SnapshotRecord]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        if record.snapshot.source_type != "sec_filing":
            continue
        payload = record.payload
        accession_number = str(payload.get("accession_number") or record.snapshot.natural_key)
        sections = payload.get("sections")
        if not isinstance(sections, dict):
            continue
        for chunk_index, (section_name, section_payload) in enumerate(sections.items()):
            if isinstance(section_payload, dict):
                text = str(section_payload.get("content") or "")
                display_name = str(section_payload.get("name") or section_name)
            else:
                text = str(section_payload)
                display_name = str(section_name)
            if not text.strip():
                continue
            chunks.append(
                {
                    "text": text,
                    "section": display_name,
                    "filing_type": str(payload.get("filing_type") or "10-K"),
                    "filing_date": payload.get("filing_date"),
                    "relevance_score": 1.0,
                    "chunk_id": f"{accession_number}_{section_name}_{chunk_index:03d}",
                }
            )
    return chunks


def _sentiment_result_from_records(records: list[SnapshotRecord]) -> dict:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for record in records:
        if record.snapshot.source_type != "sentiment_score":
            continue
        label = str(record.payload.get("label") or "neutral").lower()
        if label not in counts:
            label = "neutral"
        counts[label] += 1

    if not any(counts.values()):
        return {}
    if counts["positive"] > counts["negative"]:
        overall = "positive"
    elif counts["negative"] > counts["positive"]:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "neutral_count": counts["neutral"],
    }


def _state_from_evidence_records(
    ticker: str,
    records: list[SnapshotRecord],
    *,
    include_filing_analysis: bool,
    include_news_sentiment: bool,
    max_news_articles: int,
) -> AgentState:
    company_name = _company_name_from_records(ticker, records)
    state = create_initial_state(
        ticker,
        company_name,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )
    state["news_articles"] = _news_articles_from_records(records, max_news_articles)
    state["stock_data"] = _stock_data_from_records(records)
    if state["stock_data"].get("company_name"):
        state["company_name"] = str(state["stock_data"]["company_name"])
    if include_filing_analysis:
        state["filing_chunks"] = _filing_chunks_from_records(records)
    if include_news_sentiment:
        state["sentiment_result"] = _sentiment_result_from_records(records)
    return state


async def _run_frozen_evidence(
    ticker: str,
    records: list[SnapshotRecord],
    *,
    include_filing_analysis: bool,
    include_news_sentiment: bool,
    max_news_articles: int,
) -> tuple[AgentState, float]:
    state = _state_from_evidence_records(
        ticker,
        records,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )
    t0 = time.perf_counter()
    draft_update = await draft_memo_node(state)
    merged = AgentState(**{**state, **draft_update})
    verify_update = await verify_memo_node(merged)
    final = AgentState(**{**merged, **verify_update})
    run_ms = (time.perf_counter() - t0) * 1000.0
    return final, run_ms


def _score_final_state(
    ticker: str,
    final: AgentState,
    *,
    run_ms: float,
    checker: str,
    min_similarity: float,
) -> dict:
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
        "threshold_sensitivity": _threshold_sensitivity(gr["claims"]),
        "borderline_claims": _borderline_claims(gr["claims"], min_similarity),
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


async def evaluate_single_run(agent, ticker, *, include_filing_analysis, include_news_sentiment,
                              max_news_articles, checker: str, min_similarity: float,
                              evidence_records_by_ticker: dict[str, list[SnapshotRecord]] | None = None) -> dict:
    if evidence_records_by_ticker is None:
        state = create_initial_state(
            ticker, None,
            include_filing_analysis=include_filing_analysis,
            include_news_sentiment=include_news_sentiment,
            max_news_articles=max_news_articles,
        )
        t0 = time.perf_counter()
        final = await agent.ainvoke(state, config={})
        run_ms = (time.perf_counter() - t0) * 1000.0
    else:
        replay_ticker = ticker.upper()
        records = evidence_records_by_ticker.get(replay_ticker)
        if not records:
            raise ValueError(f"evidence release has no snapshots for ticker {replay_ticker}")
        final, run_ms = await _run_frozen_evidence(
            replay_ticker,
            records,
            include_filing_analysis=include_filing_analysis,
            include_news_sentiment=include_news_sentiment,
            max_news_articles=max_news_articles,
        )

    return _score_final_state(
        ticker,
        final,
        run_ms=run_ms,
        checker=checker,
        min_similarity=min_similarity,
    )


def _aggregate(runs: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in runs]
    n = len(runs)
    keys = ["grounded_claim_rate", "citation_coverage_rate", "total_claims", "cited_claims",
            "grounded_claims", "orphan_citations", "evidence_count", "filing_chunks",
            "news_articles", "memo_words", "run_ms", "borderline_claims"]
    out = {"n_runs": n,
           "pass_rate": sum(1 for r in runs if r["passed"]) / n if n else 0.0,
           "fatal_error_rate": sum(1 for r in runs if r["fatal_error"]) / n if n else 0.0}
    for k in keys:
        out[k] = _stats(col(k))
    # Mean grounded rate per threshold across runs -- answers "how fragile is the
    # 0.45 default" from this batch without a separate sweep run.
    if runs:
        thresholds = runs[0]["threshold_sensitivity"].keys()
        out["threshold_sensitivity"] = {
            t: statistics.fmean(r["threshold_sensitivity"][t] for r in runs) for t in thresholds
        }
    else:
        out["threshold_sensitivity"] = {}
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
    parser.add_argument("--evidence-release", help="Replay a frozen evidence release (<name:version>)")
    parser.add_argument("--registry-root", default="artifacts/dataops")
    args = parser.parse_args()

    include_filing_analysis = not args.no_filings
    include_news_sentiment = not args.no_sentiment
    evidence_bundle: EvidenceReplayBundle | None = None
    if args.evidence_release:
        evidence_bundle = load_evidence_release_records(
            args.evidence_release,
            registry_root=args.registry_root,
        )
    agent = None if evidence_bundle else create_agent()

    runs: list[dict] = []
    for ticker in args.tickers:
        for i in range(args.repeats):
            r = await evaluate_single_run(
                agent, ticker,
                include_filing_analysis=include_filing_analysis,
                include_news_sentiment=include_news_sentiment,
                max_news_articles=args.max_news,
                checker=args.checker, min_similarity=args.min_similarity,
                evidence_records_by_ticker=(
                    evidence_bundle.records_by_ticker if evidence_bundle else None
                ),
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
        "grounding_eval_version": GROUNDING_EVAL_VERSION,
        "min_similarity": args.min_similarity if args.checker == "semantic" else None,
        "evidence_release": (
            {
                "release_id": evidence_bundle.manifest.release_id,
                "dataset_name": evidence_bundle.manifest.dataset_name,
                "dataset_version": evidence_bundle.manifest.dataset_version,
                "snapshot_ids": evidence_bundle.manifest.snapshot_ids,
                "artifact_uri": evidence_bundle.manifest.artifact_uri,
            }
            if evidence_bundle
            else None
        ),
        "thresholds": {"min_citation_coverage": MIN_CITATION_COVERAGE, "min_grounded_claim_rate": MIN_GROUNDED_CLAIM_RATE},
        "config": {"tickers": args.tickers, "repeats": args.repeats,
                   "include_filing_analysis": include_filing_analysis,
                   "include_news_sentiment": include_news_sentiment, "max_news_articles": args.max_news,
                   "evidence_release": args.evidence_release, "registry_root": args.registry_root},
        "overall": overall,
        "error_steps": dict(error_steps),
        "per_ticker": {t: _aggregate([r for r in runs if r["ticker"] == t]) for t in args.tickers},
        "raw_runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / next_baseline_filename(RESULTS_DIR, model_id, settings.llm.temperature)
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
