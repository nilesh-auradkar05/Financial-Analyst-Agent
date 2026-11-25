"""Report what the filings step's topic queries retrieve at candidate pools of 15 and 50.

Runs the app's four topic queries for each ticker through `search_reranked` on one index, keeping
the top 3 per pool, and reports the pools' agreement and how often the top 3 sit in the topic's
expected section (docs/test-plan.md §25, docs/sprint-plan.md FR.0 and FR.1). No chunk text is kept.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.agents.graph import FILING_TOPIC_QUERIES
from app.components.retrieval.embeddings import embedding_identifier
from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.reranked_hybrid_retrieve import (
    FoundryCohereReranker,
    RerankerLike,
    search_reranked,
)

POOLS = (15, 50)
TOP_K = 3
DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA")
# Topic, in FILING_TOPIC_QUERIES order, and the section its passages should come from.
TOPICS = (
    ("overview", "business"),
    ("risk", "risk_factors"),
    ("financial_performance", "md&a"),
    ("competition", "business"),
)


def _company_name(store: Any, ticker: str) -> str:
    """The indexed name of the ticker's top dense hit, else the ticker (as the filings step)."""
    hits = store.search_by_ticker(ticker, ticker=ticker, n_results=1).chunks
    names = (chunk.metadata.get("company_name") for chunk in hits)
    return next((str(name) for name in names if name), ticker)


def run_app_filings_workload(
    store: Any,
    tickers: Sequence[str],
    *,
    reranker: RerankerLike | None = None,
    collection: str,
) -> dict[str, Any]:
    reranker = reranker or FoundryCohereReranker()
    cases: list[dict[str, Any]] = []
    for ticker in tickers:
        company = _company_name(store, ticker)
        for template, (topic, section) in zip(FILING_TOPIC_QUERIES, TOPICS, strict=True):
            query = template.format(company=company)
            case: dict[str, Any] = {
                "ticker": ticker,
                "topic": topic,
                "query": query,
                "expected_section": section,
            }
            for pool in POOLS:
                result = search_reranked(
                    store,
                    query=query,
                    ticker=ticker,
                    top_k=TOP_K,
                    candidates="dense",
                    candidate_pool=pool,
                    reranker=reranker,
                )
                assert result.trace is not None
                case[str(pool)] = {
                    "top3": [
                        {
                            "chunk_id": chunk.id,
                            "section_key": chunk.metadata.get("section_key"),
                            "rerank_score": chunk.metadata["reranker_score"],
                        }
                        for chunk in result.chunks
                    ],
                    "trace": result.trace.model_dump(),
                }
            kept = [{row["chunk_id"] for row in case[str(pool)]["top3"]} for pool in POOLS]
            case["top3_agreement"] = len(kept[0] & kept[1]) / TOP_K
            cases.append(case)

    section_hit_rate = {}
    for pool in POOLS:
        rows = [
            (case["expected_section"], row) for case in cases for row in case[str(pool)]["top3"]
        ]
        section_hit_rate[str(pool)] = (
            sum(row["section_key"] == section for section, row in rows) / len(rows)
            if rows
            else None
        )
    return {
        "provenance": {
            "collection": collection,
            "embedding_model": embedding_identifier(),
            "reranker": reranker.identity,
        },
        "cases": cases,
        "summary": {
            "mean_top3_agreement": (
                sum(case["top3_agreement"] for case in cases) / len(cases) if cases else None
            ),
            "section_hit_rate": section_hit_rate,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the app's filings queries at pools 15 and 50."
    )
    parser.add_argument("--collection", required=True, help="OpenSearch index to search.")
    parser.add_argument("--ticker", action="append", dest="tickers", help="Repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON to write.")
    args = parser.parse_args(argv)

    store = OpenSearchVectorStore(index_name=args.collection)
    if store.count == 0:
        raise SystemExit(f"Collection {args.collection!r} is empty; build the index first.")
    report = run_app_filings_workload(
        store, args.tickers or DEFAULT_TICKERS, collection=args.collection
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved app filings workload report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
