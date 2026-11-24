"""Trace: docs/test-plan.md §25, "Retrieval trace" (FR.0, 2026-11-24); docs/sprint-plan.md "FR.0".

`search_reranked` returns a `SearchResult` whose `trace` records the query, ticker, filing type,
candidate source, the requested pool and, per candidate in store order, its dense rank and score
and its rerank rank and score; the derived fields are the dense score gap (rank 1 minus rank
min(15, n)), the sections in the pool and the dense ranks of the returned chunks. The provider is
the loopback Foundry rerank fake or an in-test scorer; the store is the in-memory fake. Expected
traces are computed in tests/unit/trace_support.py from the store's own output and the scorer.
The trace types are looked up on the module at run time (tests/rerank_api.py), so a missing one
fails its own test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.components.retrieval.vector_store import SearchResult
from tests.rerank_api import rerank_api
from tests.unit.fake_foundry import relevance
from tests.unit.fake_retrieval_store import FakeRetrievalStore
from tests.unit.trace_support import (
    GAP_RANK,
    QUERY,
    assert_trace_equals,
    candidates_of,
    expected_trace,
    ranked_store,
)

SOURCES = ["dense", "hybrid"]
DEFAULT_POOL = 50
TRACE_FIELDS = {
    "query",
    "ticker",
    "filing_type",
    "candidate_source",
    "candidate_pool",
    "candidates",
    "dense_score_gap",
    "pool_sections",
    "kept_dense_ranks",
}


class ShortestTextReranker:
    """Prefers short texts: a signal the Foundry fake's word overlap does not share."""

    identity = "test:shortest-text"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 / len(document) for document in documents]


class ConstantReranker:
    identity = "test:constant"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.5 for _ in documents]


def _search(store: FakeRetrievalStore, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "query": QUERY,
        "ticker": "AAPL",
        "filing_type": "10-K",
        "top_k": 3,
        "candidates": "dense",
    }
    return rerank_api.search_reranked(store, **{**arguments, **overrides})


def _expect(store: FakeRetrievalStore, scorer: Any = relevance, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "source": "dense",
        "pool": DEFAULT_POOL,
        "top_k": 3,
    }
    return expected_trace(store, QUERY, scorer, **{**arguments, **overrides})


def test_the_stand_in_store_ranks_dense_and_hybrid_candidates_differently() -> None:
    """Guards the source-parametrized tests below against passing vacuously."""
    store = ranked_store(20)
    ids = {
        source: [
            chunk.id
            for chunk in candidates_of(
                store, QUERY, ticker="AAPL", filing_type="10-K", source=source, pool=DEFAULT_POOL
            )
        ]
        for source in SOURCES
    }

    assert sorted(ids["dense"]) == sorted(ids["hybrid"])
    assert ids["dense"] != ids["hybrid"]


@pytest.mark.parametrize("source", SOURCES)
def test_the_trace_records_every_candidate_with_its_dense_and_rerank_position(
    foundry_rerank, source: str
) -> None:
    """test-plan §25: Retrieval trace (query, ticker, filing type, candidate source, requested
    pool; per candidate in store order the chunk id, section, dense rank and score, rerank rank
    and score) - with 20 candidates and K = 3 every candidate still gets a rerank rank."""
    store = ranked_store(20)

    result = _search(store, candidates=source)

    expected = _expect(store, source=source)
    assert isinstance(result.trace, rerank_api.RetrievalTrace)
    assert len(result.trace.candidates) == 20
    assert sorted(c.rerank_rank for c in result.trace.candidates) == list(range(1, 21))
    assert_trace_equals(result.trace.model_dump(), expected)


def test_a_reranker_that_scores_by_text_length_sets_the_rerank_rank_and_score() -> None:
    """test-plan §25: Retrieval trace (`rerank_rank` and `rerank_score` come from the reranker
    that scored the candidates, whatever it is)."""
    store = ranked_store(20)
    reranker = ShortestTextReranker()

    result = _search(store, reranker=reranker)

    expected = _expect(store, scorer=lambda _query, text: 1.0 / len(text))
    assert_trace_equals(result.trace.model_dump(), expected)
    assert [c.rerank_rank for c in result.trace.candidates] != [
        c.dense_rank for c in result.trace.candidates
    ]


@pytest.mark.parametrize("source", SOURCES)
def test_equal_rerank_scores_keep_candidate_order_in_the_rerank_rank(source: str) -> None:
    """test-plan §25: Retrieval trace (equal scores keep candidate order in `rerank_rank`)."""
    result = _search(ranked_store(20), candidates=source, reranker=ConstantReranker())

    assert [c.rerank_rank for c in result.trace.candidates] == list(range(1, 21))
    assert [c.dense_rank for c in result.trace.candidates] == list(range(1, 21))
    assert result.trace.kept_dense_ranks == [1, 2, 3]


def test_dense_rank_is_the_candidate_rank_of_the_returned_chunks_and_names_them(
    foundry_rerank,
) -> None:
    """test-plan §25: Retrieval trace (`dense_rank` equals the chunk's `candidate_rank`;
    `kept_dense_ranks` = the dense rank of each returned chunk, in returned order)."""
    result = _search(ranked_store(20), top_k=4)

    by_id = {c.chunk_id: c for c in result.trace.candidates}
    assert len(result.chunks) == 4
    for chunk in result.chunks:
        assert by_id[chunk.id].dense_rank == chunk.metadata["candidate_rank"]
        assert by_id[chunk.id].rerank_score == pytest.approx(chunk.metadata["reranker_score"])
    assert result.trace.kept_dense_ranks == [by_id[chunk.id].dense_rank for chunk in result.chunks]


def test_the_returned_chunks_are_the_reranked_top_k_whatever_the_trace_holds(
    foundry_rerank,
) -> None:
    """test-plan §25: Retrieval trace (the returned chunks are unchanged by tracing): ids and
    order are the reranker's top K of the candidates, the trace names the same chunks."""
    store = ranked_store(20)

    result = _search(store, top_k=5)

    expected = _expect(store, top_k=5)
    assert result.trace is not None
    assert [chunk.id for chunk in result.chunks] == expected["kept_ids"]
    first_five = [
        c.chunk_id
        for c in sorted(result.trace.candidates, key=lambda candidate: candidate.rerank_rank)[:5]
    ]
    assert first_five == expected["kept_ids"]
    for chunk in result.chunks:
        stored = next(c for c in store.points if c.id == chunk.id)
        assert chunk.text == stored.text
        assert {k: chunk.metadata[k] for k in stored.metadata} == stored.metadata


@pytest.mark.parametrize("count", [20, 16, 15, 5, 2])
def test_the_gap_is_dense_rank_one_minus_rank_fifteen_or_the_last_candidate(
    foundry_rerank, count: int
) -> None:
    """test-plan §25: Retrieval trace (`dense_score_gap` = dense score at rank 1 minus dense score
    at rank min(15, n)): with more than 15 candidates it reads rank 15, not the last one."""
    store = ranked_store(count)
    chunks = candidates_of(
        store, QUERY, ticker="AAPL", filing_type="10-K", source="dense", pool=DEFAULT_POOL
    )
    assert len(chunks) == count

    result = _search(store)

    reads = min(GAP_RANK, count)
    assert result.trace.dense_score_gap == pytest.approx(
        chunks[0].relevance_score - chunks[reads - 1].relevance_score
    )
    if count > GAP_RANK:
        assert chunks[GAP_RANK - 1].relevance_score != chunks[-1].relevance_score
        assert result.trace.dense_score_gap != pytest.approx(
            chunks[0].relevance_score - chunks[-1].relevance_score
        )


@pytest.mark.parametrize("count", [1, 0])
def test_the_gap_is_absent_with_fewer_than_two_candidates(foundry_rerank, count: int) -> None:
    """test-plan §25: Retrieval trace (the gap is absent when fewer than 2 candidates)."""
    store = ranked_store(count)

    result = _search(store)

    assert isinstance(result.trace, rerank_api.RetrievalTrace)
    assert len(result.trace.candidates) == count
    assert result.trace.dense_score_gap is None
    assert result.trace.model_dump()["dense_score_gap"] is None


def test_the_trace_counts_the_sections_of_the_candidate_pool_only(foundry_rerank) -> None:
    """test-plan §25: Retrieval trace (`pool_sections` = candidate count per section; the
    requested pool is recorded): with a pool of 15 over 20 chunks the five beyond it are not
    counted."""
    store = ranked_store(20)

    result = _search(store, candidate_pool=15)

    expected = _expect(store, pool=15)
    assert result.trace.candidate_pool == 15
    assert len(result.trace.candidates) == 15
    assert result.trace.pool_sections == expected["pool_sections"]
    assert sum(result.trace.pool_sections.values()) == 15
    whole = _expect(store)["pool_sections"]
    assert result.trace.pool_sections != whole


def test_the_default_pool_is_recorded_as_requested_even_when_fewer_candidates_exist(
    foundry_rerank,
) -> None:
    """test-plan §25: Retrieval trace (the requested candidate pool, not the number found)."""
    result = _search(ranked_store(5))

    assert result.trace.candidate_pool == rerank_api.RERANK_CANDIDATE_POOL == DEFAULT_POOL
    assert len(result.trace.candidates) == 5


def test_a_search_without_a_filing_type_records_none(foundry_rerank) -> None:
    """test-plan §25: Retrieval trace (the filing type searched under, absent when unfiltered)."""
    store = ranked_store(6)

    result = _search(store, filing_type=None)

    assert result.trace.filing_type is None
    assert_trace_equals(result.trace.model_dump(), _expect(store, filing_type=None))


def test_kept_dense_ranks_has_one_entry_per_returned_chunk(foundry_rerank) -> None:
    """test-plan §25: Retrieval trace (`kept_dense_ranks`; fewer candidates than K returns all)."""
    store = ranked_store(5)

    result = _search(store, top_k=10)

    assert len(result.chunks) == 5
    assert result.trace.kept_dense_ranks == _expect(store, top_k=10)["kept_dense_ranks"]
    assert sorted(result.trace.kept_dense_ranks) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("source", SOURCES)
def test_no_candidates_gives_a_trace_without_candidates_and_no_rerank_request(
    foundry_rerank, source: str
) -> None:
    """test-plan §25: Retrieval trace (no candidates -> a trace with no candidates, no gap, and no
    rerank request)."""
    result = _search(ranked_store(5), ticker="NVDA", candidates=source)

    assert result.chunks == []
    assert isinstance(result.trace, rerank_api.RetrievalTrace)
    assert result.trace.candidates == []
    assert result.trace.dense_score_gap is None
    assert result.trace.pool_sections == {}
    assert result.trace.kept_dense_ranks == []
    assert (result.trace.query, result.trace.ticker) == (QUERY, "NVDA")
    assert result.trace.candidate_source == source
    assert foundry_rerank.requests == []


def test_the_dumped_trace_has_the_derived_fields_and_no_chunk_text(foundry_rerank) -> None:
    """test-plan §25: Retrieval trace (derived fields are part of the dump; the trace holds no
    chunk text)."""
    store = ranked_store(20)

    result = _search(store)

    dumped = result.trace.model_dump()
    assert set(dumped) == TRACE_FIELDS
    assert all(
        set(candidate)
        == {"chunk_id", "section_key", "dense_rank", "dense_score", "rerank_rank", "rerank_score"}
        for candidate in dumped["candidates"]
    )
    text = json.dumps(dumped)
    for point in store.points:
        assert point.text not in text
    assert dumped["kept_dense_ranks"] == result.trace.kept_dense_ranks
    assert dumped["pool_sections"] == result.trace.pool_sections


def test_the_trace_models_forbid_extra_fields() -> None:
    """test-plan §25: Retrieval trace (typed record: unknown fields such as chunk text are
    rejected)."""
    candidate = {
        "chunk_id": "chunk-00",
        "section_key": "business",
        "dense_rank": 1,
        "dense_score": 0.9,
        "rerank_rank": 1,
        "rerank_score": 0.5,
    }
    rerank_api.TraceCandidate(**candidate)

    with pytest.raises(ValidationError):
        rerank_api.TraceCandidate(**candidate, text="chunk text")

    assert rerank_api.RetrievalTrace.model_config.get("extra") == "forbid"
    assert rerank_api.TraceCandidate.model_config.get("extra") == "forbid"


def test_every_other_search_leaves_the_trace_empty() -> None:
    """test-plan §25: Retrieval trace (every other search leaves `trace` empty)."""
    store = ranked_store(5)
    results: list[Any] = [
        store.search(QUERY),
        store.search_hybrid(QUERY),
        store.search_by_ticker(QUERY, "AAPL"),
        SearchResult(query=QUERY, chunks=[]),
    ]

    assert [result.trace for result in results] == [None, None, None, None]
