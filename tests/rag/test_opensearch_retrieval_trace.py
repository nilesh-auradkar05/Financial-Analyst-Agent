"""Trace: docs/test-plan.md §25, "Retrieval trace on OpenSearch" (FR.0, 2026-11-24, C).

On the OpenSearch store, the trace of `search_reranked` has as candidates the store's `search()`
top `candidate_pool` for the query and filters, in the same order with the same `relevance_score`,
and its kept dense ranks name the returned chunks. Runs against the local OpenSearch test
container; vectors come from the 16-d hashed bag-of-words fake embedder, the reranker is an
in-test scorer at the provider boundary that computes its scores from the texts. Any other search
of the store leaves `trace` empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.rerank_api import rerank_api

QUERY = "supply chain disruption risk"
TOPIC_WORDS = ("supply", "chain", "disruption", "risk", "currency", "regulation")
SECTIONS = ("business", "risk_factors", "md&a")


class OverlapReranker:
    """Scores a document by how many of the query's words it holds; ties keep candidate order."""

    identity = "test:query-word-overlap"

    def score(self, query: str, documents: list[str]) -> list[float]:
        wanted = set(query.lower().split())
        return [len(wanted & set(document.lower().split())) / len(wanted) for document in documents]


def _document(number: int, ticker: str, filing_type: str) -> IndexDocument:
    words = " ".join(TOPIC_WORDS[: number % len(TOPIC_WORDS)])
    section = SECTIONS[number % len(SECTIONS)]
    chunk_id = f"{ticker}-{filing_type}-{number:02d}"
    return IndexDocument(
        id=chunk_id,
        text=f"{ticker} {filing_type} passage {number} {words}",
        metadata={
            "ticker": ticker,
            "company_name": ticker,
            "filing_type": filing_type,
            "filing_date": "2025-01-31",
            "accession_number": f"{ticker}-{filing_type}",
            "source_url": f"https://www.sec.gov/Archives/{ticker}-{filing_type}",
            "section": section,
            "section_name": section,
            "section_key": section,
            "chunk_id": chunk_id,
            "chunk_index": number,
        },
    )


@pytest.fixture
def indexed(index_name: str, make_store: Any) -> Any:
    """Twenty-four AAPL 10-K chunks, plus AAPL 10-Q and MSFT 10-K chunks that must never be
    candidates for an AAPL 10-K search."""
    store = make_store(index_name)
    store.add_documents(
        [_document(n, "AAPL", "10-K") for n in range(24)]
        + [_document(n, "AAPL", "10-Q") for n in range(6)]
        + [_document(n, "MSFT", "10-K") for n in range(6)]
    )
    return store


def _search(store: Any, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "query": QUERY,
        "ticker": "AAPL",
        "filing_type": "10-K",
        "top_k": 4,
        "candidates": "dense",
        "reranker": OverlapReranker(),
    }
    return rerank_api.search_reranked(store, **{**arguments, **overrides})


@pytest.mark.parametrize("pool", [8, 15, 50])
def test_trace_candidates_are_the_stores_top_pool_in_the_same_order_and_score(
    indexed: Any, pool: int
) -> None:
    """test-plan §25: Retrieval trace on OpenSearch (the trace's candidates are the store's
    `search()` top `candidate_pool` for the query and filters, in the same order with the same
    `relevance_score`)."""
    stored = indexed.search(
        QUERY, filters=SearchFilters(ticker="AAPL", filing_type="10-K"), n_results=pool
    ).chunks
    assert len(stored) == min(pool, 24)

    result = _search(indexed, candidate_pool=pool)

    trace = result.trace
    assert isinstance(trace, rerank_api.RetrievalTrace)
    assert [c.chunk_id for c in trace.candidates] == [chunk.id for chunk in stored]
    assert [c.dense_rank for c in trace.candidates] == list(range(1, len(stored) + 1))
    assert [c.dense_score for c in trace.candidates] == pytest.approx(
        [chunk.relevance_score for chunk in stored]
    )
    assert [c.section_key for c in trace.candidates] == [
        chunk.metadata["section_key"] for chunk in stored
    ]
    assert (trace.query, trace.ticker, trace.filing_type) == (QUERY, "AAPL", "10-K")
    assert trace.candidate_source == "dense"
    assert trace.candidate_pool == pool
    assert not any(c.chunk_id.startswith(("MSFT", "AAPL-10-Q")) for c in trace.candidates)
    if pool < 24:
        assert len(trace.candidates) == pool


def test_kept_dense_ranks_name_the_returned_chunks_and_the_gap_reads_the_store_scores(
    indexed: Any,
) -> None:
    """test-plan §25: Retrieval trace on OpenSearch (its kept dense ranks name the returned
    chunks; the derived gap is the store's score at rank 1 minus rank 15)."""
    stored = indexed.search(
        QUERY, filters=SearchFilters(ticker="AAPL", filing_type="10-K"), n_results=50
    ).chunks
    dense_rank = {chunk.id: rank for rank, chunk in enumerate(stored, start=1)}

    result = _search(indexed)

    trace = result.trace
    assert len(result.chunks) == 4
    assert trace.kept_dense_ranks == [dense_rank[chunk.id] for chunk in result.chunks]
    assert trace.kept_dense_ranks == [chunk.metadata["candidate_rank"] for chunk in result.chunks]
    assert trace.dense_score_gap == pytest.approx(
        stored[0].relevance_score - stored[14].relevance_score
    )
    assert sum(trace.pool_sections.values()) == 24
    reranked = sorted(trace.candidates, key=lambda c: (-c.rerank_score, c.dense_rank))
    assert [c.chunk_id for c in reranked[:4]] == [chunk.id for chunk in result.chunks]
    assert [c.rerank_rank for c in reranked] == list(range(1, 25))


def test_a_plain_search_of_the_store_leaves_the_trace_empty(indexed: Any) -> None:
    """test-plan §25: Retrieval trace (every other search leaves `trace` empty), on the real
    store."""
    filters = SearchFilters(ticker="AAPL", filing_type="10-K")

    results = [
        indexed.search(QUERY, filters=filters, n_results=5),
        indexed.search_hybrid(QUERY, filters=filters, n_results=5),
        indexed.search_by_ticker(QUERY, "AAPL", n_results=5),
    ]

    assert [result.trace for result in results] == [None, None, None]
