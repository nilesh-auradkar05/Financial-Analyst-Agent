"""Expected retrieval traces, computed from a store and a scorer (test-plan §25, FR.0).

Every expected value is derived from what the store's own `search()` / `search_hybrid()` return
(chunk order and `relevance_score`) and from a scorer standing in for the reranker; nothing here
is a typed-in answer. The scorer is called as `scorer(query, text) -> float`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from prometheus_client.parser import text_string_to_metric_families

from app.components.retrieval.vector_store import IndexDocument, RetrievedChunk, SearchFilters
from tests.unit.fake_retrieval_store import FakeRetrievalStore, make_point

GAP_RANK = 15
SECTION_CYCLE = ("business", "risk_factors", "md&a")
QUERY_WORDS = ("supply", "chain", "supplier", "risks")
QUERY = " ".join(QUERY_WORDS)

Scorer = Callable[[str, str], float]


def ranked_store(count: int, ticker: str = "AAPL", filing_type: str = "10-K") -> FakeRetrievalStore:
    """`count` chunks of one filing in dense order, each carrying a section and a varying share of
    the query words, so the rerank order differs from the dense order and has ties."""
    return FakeRetrievalStore(
        [
            make_point(
                f"chunk-{number:02d}",
                f"note {number} " + " ".join(QUERY_WORDS[: number % 5]),
                SECTION_CYCLE[number % len(SECTION_CYCLE)],
                ticker=ticker,
                filing_type=filing_type,
            )
            for number in range(count)
        ]
    )


def candidates_of(
    store: Any,
    query: str,
    *,
    ticker: str,
    filing_type: str | None,
    source: str,
    pool: int,
) -> list[RetrievedChunk]:
    search = store.search if source == "dense" else store.search_hybrid
    chunks: list[RetrievedChunk] = search(
        query, filters=SearchFilters(ticker=ticker, filing_type=filing_type), n_results=pool
    ).chunks
    return chunks


def expected_trace(
    store: Any,
    query: str,
    scorer: Scorer,
    *,
    ticker: str,
    filing_type: str | None,
    source: str,
    pool: int,
    top_k: int,
) -> dict[str, Any]:
    """The trace `search_reranked` must record: candidates in store order with dense score from the
    store, rerank rank by scorer (equal scores in candidate order), and the derived fields."""
    chunks = candidates_of(
        store, query, ticker=ticker, filing_type=filing_type, source=source, pool=pool
    )
    scores = [scorer(query, chunk.text) for chunk in chunks]
    order = sorted(range(len(chunks)), key=lambda index: -scores[index])
    rerank_rank = {index: place for place, index in enumerate(order, start=1)}
    candidates = [
        {
            "chunk_id": chunk.id,
            "section_key": chunk.metadata.get("section_key"),
            "dense_rank": index + 1,
            "dense_score": chunk.relevance_score,
            "rerank_rank": rerank_rank[index],
            "rerank_score": scores[index],
        }
        for index, chunk in enumerate(chunks)
    ]
    gap = None
    if len(chunks) >= 2:
        gap = chunks[0].relevance_score - chunks[min(GAP_RANK, len(chunks)) - 1].relevance_score
    return {
        "query": query,
        "ticker": ticker,
        "filing_type": filing_type,
        "candidate_source": source,
        "candidate_pool": pool,
        "candidates": candidates,
        "dense_score_gap": gap,
        "pool_sections": dict(Counter(c["section_key"] for c in candidates)),
        "kept_dense_ranks": [index + 1 for index in order[:top_k]],
        "kept_ids": [chunks[index].id for index in order[:top_k]],
    }


def assert_trace_equals(
    actual: dict[str, Any], expected: dict[str, Any], *, filing_type: bool = True
) -> None:
    """`actual` is a trace's `model_dump()`; floats compare approximately."""
    scalar = ["query", "ticker", "candidate_source", "candidate_pool", "pool_sections"]
    scalar += ["kept_dense_ranks"]
    if filing_type:
        scalar.append("filing_type")
    for name in scalar:
        assert actual[name] == expected[name], name
    if expected["dense_score_gap"] is None:
        assert actual["dense_score_gap"] is None
    else:
        assert actual["dense_score_gap"] == pytest.approx(expected["dense_score_gap"])
    assert len(actual["candidates"]) == len(expected["candidates"])
    for got, want in zip(actual["candidates"], expected["candidates"], strict=True):
        assert set(got) == set(want)
        for name in ("chunk_id", "section_key", "dense_rank", "rerank_rank"):
            assert got[name] == want[name], (want["chunk_id"], name)
        assert got["dense_score"] == pytest.approx(want["dense_score"])
        assert got["rerank_score"] == pytest.approx(want["rerank_score"])


FILING_TEMPLATES = (
    "{company} business description overview",
    "{company} risk factors challenges",
    "{company} financial performance revenue",
    "{company} competition market position",
)
APP_POOL = 15
APP_TOP_K = 3
GAP_HISTOGRAM = "retrieval_dense_score_gap"
RANK_HISTOGRAM = "retrieval_kept_dense_rank"


def filing_points(ticker: str, company: str) -> list[IndexDocument]:
    """One company's filing as the filings step sees it: three plain passages first (what a plain
    dense search returns), the passages that answer the topic queries next, then thirty filler
    passages, so a pool of 15 is full and the passages beyond it are never candidates."""
    texts = [f"{ticker} plain passage {number}." for number in range(3)]
    texts += [f"Business description overview of segment {number}." for number in range(4)]
    texts += [
        "Risk factors in manufacturing.",
        "Financial performance in the fourth quarter.",
        "Competition from new entrants.",
    ]
    texts += [f"Widgets are catalogued under note {number}." for number in range(30)]
    return [
        IndexDocument(
            id=f"{ticker}-{number:02d}",
            text=text,
            metadata={
                "ticker": ticker,
                "filing_type": "10-K",
                "section_key": SECTION_CYCLE[number % len(SECTION_CYCLE)],
                "section_name": SECTION_CYCLE[number % len(SECTION_CYCLE)],
                "company_name": company,
                "filing_date": "2026-01-01",
                "fetched_at": "2026-11-20T10:00:00+00:00",
            },
        )
        for number, text in enumerate(texts)
    ]


def expected_filing_traces(
    store: Any, ticker: str, company: str, scorer: Scorer
) -> list[dict[str, Any]]:
    """The four traces a live filings step records, in query order."""
    return [
        expected_trace(
            store,
            template.format(company=company),
            scorer,
            ticker=ticker,
            filing_type=None,
            source="dense",
            pool=APP_POOL,
            top_k=APP_TOP_K,
        )
        for template in FILING_TEMPLATES
    ]


def histogram_totals(exposition: str, family: str) -> tuple[float, float]:
    """(`_count`, `_sum`) of the `stage="filings"` series of a histogram in /metrics text."""
    totals = {"_count": 0.0, "_sum": 0.0}
    for metric in text_string_to_metric_families(exposition):
        for sample in metric.samples:
            suffix = sample.name.removeprefix(family)
            if suffix in totals and sample.labels.get("stage") == "filings":
                totals[suffix] += sample.value
    return totals["_count"], totals["_sum"]
