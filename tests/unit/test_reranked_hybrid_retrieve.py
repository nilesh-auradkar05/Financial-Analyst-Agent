"""Trace: docs/test-plan.md §25, "Rerank reorders candidates (amended S5.3, 2026-11-15; pool amended
FILINGS-LATENCY 2026-11-23)" and
"Fails closed when unconfigured" (amended S5.3).

`search_reranked` takes the store's top 50 (dense `search()` or `search_hybrid()`) under the ticker
and filing-type filters, scores them in one rerank call and keeps the K the provider scores
highest. The provider is the loopback Foundry rerank fake (tests/unit/fake_foundry.py), which
computes relevance from the request; the store is the in-memory fake of
tests/unit/fake_retrieval_store.py. Expected orders are computed here from the store's own output
and the fake's relevance function, never typed in as the answer of the code under test.
"""

from __future__ import annotations

import pytest

from app.components.retrieval.reranked_hybrid_retrieve import (
    RerankedHybridSearchUnavailableError,
)
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters
from app.config import settings
from tests.rerank_api import rerank_api
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, relevance
from tests.unit.fake_retrieval_store import (
    BUSINESS_TEXT,
    RISK_TEXT,
    FakeRetrievalStore,
    make_point,
)

QUERY = "supply chain supplier risks"
IDENTITY = f"foundry:{RERANK_DEPLOYMENT}"
SOURCES = ["dense", "hybrid"]
FILTERS = SearchFilters(ticker="AAPL", filing_type="10-K")
POOL = 50


def _scenario_store() -> FakeRetrievalStore:
    """Dense order is the order below. Provider relevance for the query (share of its four words a
    text holds): risk-1 1.0, mid-1 0.75, tie-a and tie-b 0.25, business-1 0. The last two points
    would score 1.0 but belong to another company and to a 10-Q."""
    return FakeRetrievalStore(
        [
            make_point("business-1", BUSINESS_TEXT, "business"),
            make_point("tie-a", "Supply agreements renew annually.", "business"),
            make_point("mid-1", "Supplier risks to the chain are managed.", "risk_factors"),
            make_point("tie-b", "Supply forecasts guide purchasing.", "business"),
            make_point("risk-1", RISK_TEXT, "risk_factors"),
            make_point("msft-risk", RISK_TEXT, "risk_factors", ticker="MSFT"),
            make_point("aapl-10q-risk", RISK_TEXT, "risk_factors", filing_type="10-Q"),
        ]
    )


def _crowded_store() -> FakeRetrievalStore:
    """Sixty AAPL 10-K chunks. Dense rank 50 holds a half match for the query and dense rank 51 a
    perfect one; every other chunk matches no query word. The fused order puts rank 51 first."""
    points = [
        make_point(
            f"filler-{n:02d}", f"Quarterly dividend note {n} on shareholder returns.", "business"
        )
        for n in range(60)
    ]
    points[49] = make_point("dense-rank-50", "Supplier risks are reviewed.", "risk_factors")
    points[50] = make_point(
        "dense-rank-51", "Supply chain supplier risks overview.", "risk_factors"
    )
    return FakeRetrievalStore(points)


def _candidates(store: FakeRetrievalStore, source: str, filing_type: str | None = "10-K"):
    filters = SearchFilters(ticker="AAPL", filing_type=filing_type)
    search = store.search if source == "dense" else store.search_hybrid
    return search(QUERY, filters=filters, n_results=POOL).chunks


def _expected_ids(candidates: list[RetrievedChunk], top_k: int) -> list[str]:
    """Highest provider score first, equal scores in candidate order (`sorted` is stable)."""
    ranked = sorted(candidates, key=lambda chunk: -relevance(QUERY, chunk.text))
    return [chunk.id for chunk in ranked[:top_k]]


def _search(store: FakeRetrievalStore, **overrides):
    arguments = {
        "query": QUERY,
        "ticker": "AAPL",
        "filing_type": "10-K",
        "top_k": 10,
        "candidates": "dense",
    }
    return rerank_api.search_reranked(store, **{**arguments, **overrides})


def test_the_candidate_pool_is_fifty() -> None:
    """test-plan §25: Rerank reorders candidates (the store's top 50)."""
    assert rerank_api.RERANK_CANDIDATE_POOL == POOL


def test_dense_candidates_are_reranked_best_first_with_provenance(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (order, reranker_score, candidate_rank, source,
    identity; chunk id, text and other metadata as the store returned them)."""
    store = _scenario_store()

    result = _search(store, top_k=3)

    assert [chunk.id for chunk in result.chunks] == ["risk-1", "mid-1", "tie-a"]
    by_id = {
        chunk.id: chunk for chunk in store.search(QUERY, filters=FILTERS, n_results=POOL).chunks
    }
    expected_rank = {"business-1": 1, "tie-a": 2, "mid-1": 3, "tie-b": 4, "risk-1": 5}
    for chunk in result.chunks:
        stored = by_id[chunk.id]
        assert chunk.text == stored.text
        assert chunk.metadata["reranker_score"] == pytest.approx(relevance(QUERY, stored.text))
        assert isinstance(chunk.metadata["reranker_score"], float)
        assert chunk.metadata["candidate_rank"] == expected_rank[chunk.id]
        assert chunk.metadata["candidate_source"] == "dense"
        assert chunk.metadata["reranker_model"] == IDENTITY
        for name, value in stored.metadata.items():
            assert chunk.metadata[name] == value


@pytest.mark.parametrize("source", SOURCES)
def test_the_k_highest_scored_candidates_are_returned_highest_first(
    foundry_rerank, source: str
) -> None:
    """test-plan §25: Rerank reorders candidates (dense `search()` or `search_hybrid()` candidates;
    K highest by provider score; candidate_rank is the position in that source's list)."""
    store = _scenario_store()
    candidates = _candidates(store, source)

    result = _search(store, top_k=3, candidates=source)

    assert [chunk.id for chunk in result.chunks] == _expected_ids(candidates, 3)
    position = {chunk.id: rank for rank, chunk in enumerate(candidates, start=1)}
    scores = [chunk.metadata["reranker_score"] for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    for chunk in result.chunks:
        assert chunk.metadata["candidate_rank"] == position[chunk.id]
        assert chunk.metadata["candidate_source"] == source
        assert chunk.metadata["reranker_model"] == IDENTITY


def test_the_hybrid_candidate_list_ranks_differently_from_the_dense_one() -> None:
    """Guards the two tests around it against passing vacuously: the stand-in store's hybrid order
    differs from its dense order, so the candidate source shows in `candidate_rank`."""
    store = _scenario_store()

    dense = [chunk.id for chunk in _candidates(store, "dense")]
    hybrid = [chunk.id for chunk in _candidates(store, "hybrid")]

    assert sorted(dense) == sorted(hybrid)
    assert dense != hybrid


def test_equal_scores_keep_candidate_order(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (equal scores keep candidate order)."""
    result = _search(_scenario_store(), top_k=5)

    assert [chunk.id for chunk in result.chunks] == [
        "risk-1",
        "mid-1",
        "tie-a",
        "tie-b",
        "business-1",
    ]


@pytest.mark.parametrize("source", SOURCES)
def test_fewer_candidates_than_k_returns_all_of_them_reranked(foundry_rerank, source: str) -> None:
    """test-plan §25: Rerank reorders candidates (fewer than K candidates -> all, reranked)."""
    store = _scenario_store()
    candidates = _candidates(store, source)

    result = _search(store, top_k=10, candidates=source)

    assert len(candidates) == 5
    assert [chunk.id for chunk in result.chunks] == _expected_ids(candidates, 10)


@pytest.mark.parametrize("source", SOURCES)
def test_candidates_are_scoped_to_the_ticker_and_filing_type(foundry_rerank, source: str) -> None:
    """test-plan §25: Rerank reorders candidates (candidates are taken under the ticker and
    filing-type filters; a perfect match for another company or a 10-Q never appears)."""
    result = _search(_scenario_store(), top_k=10, candidates=source)

    assert "msft-risk" not in {chunk.id for chunk in result.chunks}
    assert "aapl-10q-risk" not in {chunk.id for chunk in result.chunks}
    (request,) = foundry_rerank.requests
    assert RISK_TEXT in request.body["documents"]
    assert len(request.body["documents"]) == 5


def test_without_a_filing_type_every_filing_of_the_ticker_is_a_candidate(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (the filing-type filter is the caller's)."""
    store = _scenario_store()
    candidates = _candidates(store, "dense", filing_type=None)

    result = _search(store, top_k=2, filing_type=None)

    assert {chunk.id for chunk in candidates} == {
        "business-1",
        "tie-a",
        "mid-1",
        "tie-b",
        "risk-1",
        "aapl-10q-risk",
    }
    assert [chunk.id for chunk in result.chunks] == ["risk-1", "aapl-10q-risk"]


def test_dense_candidates_are_the_stores_top_fifty_and_no_more(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (the pool is 50: the dense rank-51 perfect match
    is never scored, the rank-50 half match is)."""
    result = _search(_crowded_store(), top_k=3)

    ids = [chunk.id for chunk in result.chunks]
    assert ids[0] == "dense-rank-50"
    assert "dense-rank-51" not in ids
    assert result.chunks[0].metadata["candidate_rank"] == POOL
    (request,) = foundry_rerank.requests
    assert len(request.body["documents"]) == POOL
    assert "Supply chain supplier risks overview." not in request.body["documents"]


def test_hybrid_candidates_are_the_stores_hybrid_top_fifty(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (the hybrid pool is the store's hybrid top 50: the
    chunk dense rank 51 reaches in fused order is scored and wins)."""
    store = _crowded_store()
    candidates = _candidates(store, "hybrid")
    assert "dense-rank-51" in {chunk.id for chunk in candidates}
    assert "dense-rank-51" not in {chunk.id for chunk in _candidates(store, "dense")}

    result = _search(store, top_k=3, candidates="hybrid")

    assert [chunk.id for chunk in result.chunks] == _expected_ids(candidates, 3)
    assert result.chunks[0].id == "dense-rank-51"
    (request,) = foundry_rerank.requests
    assert len(request.body["documents"]) == POOL


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("pool", [15, 30])
def test_candidate_pool_sets_how_many_candidates_are_asked_for_and_scored(
    foundry_rerank, source: str, pool: int
) -> None:
    """test-plan §25: Rerank reorders candidates (amended FILINGS-LATENCY 2026-11-23): the top
    `candidate_pool` of the store's dense or hybrid search go to the reranker in one call, so a
    pool of 15 never scores the chunk at candidate rank 16, and each returned chunk's
    `candidate_rank` is its place inside that pool."""
    store = _crowded_store()
    candidates = (store.search if source == "dense" else store.search_hybrid)(
        QUERY, filters=FILTERS, n_results=pool
    ).chunks
    assert len(candidates) == pool

    result = _search(store, top_k=3, candidates=source, candidate_pool=pool)

    (request,) = foundry_rerank.requests
    assert sorted(request.body["documents"]) == sorted(chunk.text for chunk in candidates)
    assert [chunk.id for chunk in result.chunks] == _expected_ids(candidates, 3)
    assert all(chunk.metadata["candidate_rank"] <= pool for chunk in result.chunks)


def test_a_pool_of_fifteen_does_not_see_the_dense_rank_fifty_match(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (amended FILINGS-LATENCY 2026-11-23): with
    `candidate_pool` = 15 the half match at dense rank 50 is not a candidate; with the default pool
    it is, and wins."""
    store = _crowded_store()

    narrow = _search(store, top_k=3, candidate_pool=15)
    wide = _search(store, top_k=3)

    assert "dense-rank-50" not in [chunk.id for chunk in narrow.chunks]
    assert wide.chunks[0].id == "dense-rank-50"
    narrow_request, wide_request = foundry_rerank.requests
    assert len(narrow_request.body["documents"]) == 15
    assert len(wide_request.body["documents"]) == POOL


@pytest.mark.parametrize("source", SOURCES)
def test_all_candidates_go_to_the_reranker_in_one_request(foundry_rerank, source: str) -> None:
    """test-plan §25: Rerank reorders candidates (sends them to the reranker in one call)."""
    store = _scenario_store()
    candidates = _candidates(store, source)

    _search(store, top_k=2, candidates=source)

    (request,) = foundry_rerank.requests
    assert request.body["query"] == QUERY
    assert sorted(request.body["documents"]) == sorted(chunk.text for chunk in candidates)


@pytest.mark.parametrize("source", SOURCES)
def test_no_candidates_gives_an_empty_result_and_no_rerank_request(
    foundry_rerank, source: str
) -> None:
    """test-plan §25: Rerank reorders candidates (no candidates -> empty result, no request)."""
    result = _search(_scenario_store(), ticker="NVDA", candidates=source)

    assert result.chunks == []
    assert foundry_rerank.requests == []


class ShortestTextReranker:
    """A scorer at the provider boundary that prefers short texts, a signal no section or keyword
    prior would share. It records what it was asked."""

    identity = "test:shortest-text"

    def __init__(self) -> None:
        self.asked: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.asked.append((query, list(documents)))
        return [1.0 / len(document) for document in documents]


def test_the_order_is_the_reranker_score_alone_and_records_the_reranker_identity() -> None:
    """test-plan §25: Rerank reorders candidates (no section or keyword prior changes the order;
    the identity is the reranker's own; one scoring call)."""
    store = _scenario_store()
    reranker = ShortestTextReranker()
    candidates = _candidates(store, "dense")

    result = _search(store, top_k=4, reranker=reranker)

    shortest_first = sorted(candidates, key=lambda chunk: len(chunk.text))
    assert [chunk.id for chunk in result.chunks] == [chunk.id for chunk in shortest_first[:4]]
    assert [chunk.id for chunk in result.chunks] != _expected_ids(candidates, 4)
    for chunk in result.chunks:
        assert chunk.metadata["reranker_model"] == "test:shortest-text"
        assert chunk.metadata["reranker_score"] == pytest.approx(1.0 / len(chunk.text))
    (asked,) = reranker.asked
    assert asked[0] == QUERY
    assert sorted(asked[1]) == sorted(chunk.text for chunk in candidates)


class ConstantReranker:
    identity = "test:constant"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.5 for _ in documents]


@pytest.mark.parametrize("source", SOURCES)
def test_a_reranker_that_scores_everything_alike_leaves_candidate_order(source: str) -> None:
    """test-plan §25: Rerank reorders candidates (equal scores keep candidate order, whatever the
    candidate source)."""
    store = _scenario_store()

    result = _search(store, top_k=4, candidates=source, reranker=ConstantReranker())

    assert [chunk.id for chunk in result.chunks] == [
        chunk.id for chunk in _candidates(store, source)[:4]
    ]


MISSING = ["azure_openai_endpoint", "azure_api_key", "azure_foundry_rerank_model"]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("missing", MISSING)
def test_unconfigured_rerank_fails_closed_before_any_request(
    foundry_rerank, monkeypatch, missing: str, source: str
) -> None:
    """test-plan §25: Fails closed when unconfigured (endpoint, key or model missing; no unreranked
    fallback)."""
    monkeypatch.setattr(settings.llm, missing, None)

    with pytest.raises(RerankedHybridSearchUnavailableError):
        _search(_scenario_store(), candidates=source)

    assert foundry_rerank.requests == []


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("status", [429, 503])
def test_a_reranker_failure_raises_and_returns_no_unreranked_results(
    foundry_rerank, source: str, status: int
) -> None:
    """test-plan §25: Fails closed when unconfigured (a reranker failure also raises; no
    unreranked fallback)."""
    foundry_rerank.status = status

    with pytest.raises(RerankedHybridSearchUnavailableError) as raised:
        _search(_scenario_store(), candidates=source)

    assert "test-key" not in str(raised.value)


class BrokenReranker:
    identity = "test:broken"

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise RerankedHybridSearchUnavailableError("the reranker is down")


def test_an_injected_reranker_that_fails_is_not_replaced_by_dense_results() -> None:
    """test-plan §25: Fails closed when unconfigured (`fail_open_to_dense` removed: nothing
    stands in for a failed reranker)."""
    with pytest.raises(RerankedHybridSearchUnavailableError):
        _search(_scenario_store(), reranker=BrokenReranker())
