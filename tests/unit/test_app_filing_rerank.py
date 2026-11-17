"""Trace: docs/test-plan.md §25, "App filing retrieval reranks" and "App rerank fallback"
(S5.3 app switch, owner 2026-11-17).

The app's filing retrieval (`retrieve_filings`) reranks each topic query's dense candidates through
the hosted reranker and keeps the best `MAX_FILING_CHUNKS`; when the reranker is unavailable it
returns the plain dense results, marked as not reranked, and the run continues. The reranker is the
loopback Foundry fake (tests/unit/fake_foundry.py), which scores each document from the request it
receives; the store is the in-memory fake (tests/unit/fake_retrieval_store.py). The expected chunks
are derived from the requests the loopback server observed. No network beyond 127.0.0.1, no
credentials, no real sleeping (the `foundry_rerank` fixture runs on the fake clock).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.agents import graph
from app.agents.state import MAX_FILING_CHUNKS, create_initial_state
from app.components.retrieval.vector_store import IndexDocument, SearchFilters, SearchResult
from app.config import settings
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, FakeFoundry, relevance
from tests.unit.fake_retrieval_store import FakeRetrievalStore

TICKER = "AAPL"
COMPANY = "Acme"
IDENTITY = f"foundry:{RERANK_DEPLOYMENT}"
PER_QUERY_K = 3


def _point(chunk_id: str, text: str, ticker: str = TICKER) -> IndexDocument:
    return IndexDocument(
        id=chunk_id,
        text=text,
        metadata={
            "ticker": ticker,
            "filing_type": "10-K",
            "section_key": "business",
            "company_name": COMPANY,
        },
    )


# The dense order is the order of this list. The three plain passages come first, so a plain dense
# search returns them for every query; the passages that answer the topic queries come later.
POINTS = [
    _point("plain-1", "Hardware shipped globally."),
    _point("plain-2", "Services were launched."),
    _point("plain-3", "Widgets are sold worldwide."),
    _point("business-1", "Business description overview of the consumer segment."),
    _point("business-2", "Business description overview of the enterprise segment."),
    _point("business-3", "Business description overview of the services segment."),
    _point("business-4", "Business description overview of the retail segment."),
    _point("risk-1", "Risk factors in manufacturing."),
    _point("risk-2", "Risk in logistics."),
    _point("finance-1", "Financial performance in the fourth quarter."),
    _point("market-1", "Competition from new entrants."),
    # Another company's passage that answers every topic query; it must never be a candidate.
    _point(
        "decoy-1",
        "Acme business description overview risk factors challenges financial performance "
        "revenue competition market position.",
        ticker="MSFT",
    ),
]
DENSE_TEXTS = [point.text for point in POINTS[:3]]
DENSE_IDS = [point.id for point in POINTS[:3]]


def _store() -> Any:
    return FakeRetrievalStore(POINTS)


def _expected(queries: list[str]) -> dict[str, float]:
    """The chunks the app must keep, with the score the reranker gives each.

    Each query keeps its `PER_QUERY_K` best ticker candidates by the provider's relevance (equal
    scores in candidate order); the merged set keeps the `MAX_FILING_CHUNKS` best.
    """
    candidates = [point for point in POINTS if point.metadata["ticker"] == TICKER]
    kept_first: dict[str, float] = {}
    kept_best: dict[str, float] = {}
    for query in queries:
        ranked = sorted(candidates, key=lambda point: -relevance(query, point.text))
        for point in ranked[:PER_QUERY_K]:
            score = relevance(query, point.text)
            kept_first.setdefault(point.id, score)
            kept_best[point.id] = max(score, kept_best.get(point.id, 0.0))
    # The data must not depend on how duplicates are merged, nor on a tie at the cut-off.
    assert kept_first == kept_best
    ordered = sorted(kept_best.values(), reverse=True)
    assert ordered[MAX_FILING_CHUNKS - 1] > ordered[MAX_FILING_CHUNKS]
    best = sorted(kept_best, key=lambda chunk_id: -kept_best[chunk_id])[:MAX_FILING_CHUNKS]
    return {chunk_id: kept_best[chunk_id] for chunk_id in best}


def _use(monkeypatch: pytest.MonkeyPatch, store: object) -> None:
    monkeypatch.setattr(graph, "get_vector_store", lambda: store)


def _queries(foundry: FakeFoundry) -> list[str]:
    return [request.body["query"] for request in foundry.requests]


@pytest.mark.parametrize("company_name", ["", COMPANY], ids=["name-from-index", "name-in-state"])
async def test_filing_retrieval_keeps_the_best_five_by_reranker_score(
    foundry_rerank: FakeFoundry, monkeypatch: pytest.MonkeyPatch, company_name: str
) -> None:
    """test-plan §25: App filing retrieval reranks (the five best of the merged set by
    reranker_score, highest first, from the ticker's candidates; the company-name lookup stays
    plain dense, so only the four topic queries reach the reranker)."""
    _use(monkeypatch, _store())

    update = await graph.retrieve_sec_filings_node(create_initial_state(TICKER, company_name))

    queries = _queries(foundry_rerank)
    assert len(queries) == len(set(queries)) == 4
    assert all(COMPANY in query for query in queries)
    assert TICKER not in queries
    for request in foundry_rerank.requests:
        assert sorted(request.body["documents"]) == sorted(
            point.text for point in POINTS if point.metadata["ticker"] == TICKER
        )
    expected = _expected(queries)
    texts = {point.id: point.text for point in POINTS}
    kept = [chunk["text"] for chunk in update["filing_chunks"]]
    assert sorted(kept) == sorted(texts[chunk_id] for chunk_id in expected)
    scores = {texts[chunk_id]: score for chunk_id, score in expected.items()}
    assert [scores[text] for text in kept] == sorted(scores.values(), reverse=True)
    assert not update.get("errors")


async def test_each_kept_chunk_records_its_rerank_score_and_the_reranker(
    foundry_rerank: FakeFoundry,
) -> None:
    """test-plan §25: App filing retrieval reranks (each kept chunk carries `reranker_score`, the
    reranker identity and `reranked` = true)."""
    chunks = await graph._retrieve_filing_chunks(_store(), TICKER, COMPANY)

    assert len(foundry_rerank.requests) == 4
    expected = _expected(_queries(foundry_rerank))
    assert [chunk.id for chunk in chunks if chunk.id not in expected] == []
    assert {chunk.id for chunk in chunks} == set(expected)
    for chunk in chunks:
        assert chunk.metadata.get("reranked") is True
        assert chunk.metadata.get("reranker_model") == IDENTITY
        assert chunk.metadata.get("reranker_score") == pytest.approx(expected[chunk.id])
        assert "rerank_fallback_reason" not in chunk.metadata
    got = [chunk.metadata.get("reranker_score") for chunk in chunks]
    assert got == sorted(got, reverse=True)


async def test_rerank_keeps_three_chunks_per_query(foundry_rerank: FakeFoundry) -> None:
    """test-plan §25: App filing retrieval reranks (the app's per-query K is 3): the fourth
    equally relevant passage of the business query is not kept."""
    chunks = await graph._retrieve_filing_chunks(_store(), TICKER, COMPANY)

    kept = {chunk.id for chunk in chunks}
    assert {"business-1", "business-2", "business-3"} <= kept
    assert "business-4" not in kept
    assert "decoy-1" not in kept


# ---- fallback -------------------------------------------------------------------------------


def _unset(setting: str) -> Callable[[FakeFoundry, pytest.MonkeyPatch], None]:
    def break_it(_foundry: FakeFoundry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.llm, setting, None)

    return break_it


def _fail_with(status: int) -> Callable[[FakeFoundry, pytest.MonkeyPatch], None]:
    def break_it(foundry: FakeFoundry, _monkeypatch: pytest.MonkeyPatch) -> None:
        foundry.status = status

    return break_it


def _malformed(foundry: FakeFoundry, _monkeypatch: pytest.MonkeyPatch) -> None:
    foundry.raw_reply = b"<html>502 Bad Gateway</html>"


UNCONFIGURED = {
    "no-endpoint": _unset("azure_openai_endpoint"),
    "no-key": _unset("azure_api_key"),
    "no-model": _unset("azure_foundry_rerank_model"),
}
PROVIDER_FAILURES = {
    "http-429": _fail_with(429),
    "http-500": _fail_with(500),
    "http-503": _fail_with(503),
    "malformed-reply": _malformed,
}
BREAKS = {**UNCONFIGURED, **PROVIDER_FAILURES}


@pytest.fixture
def broken_reranker(
    request: pytest.FixtureRequest, foundry_rerank: FakeFoundry, monkeypatch: pytest.MonkeyPatch
) -> FakeFoundry:
    BREAKS[request.param](foundry_rerank, monkeypatch)
    return foundry_rerank


@pytest.mark.parametrize("broken_reranker", BREAKS, indirect=True)
async def test_an_unavailable_reranker_leaves_the_run_with_the_dense_filings(
    broken_reranker: FakeFoundry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test-plan §25: App rerank fallback (the step returns the plain dense results in their old
    order, the run keeps its filings and records no filing error)."""
    _use(monkeypatch, _store())

    update = await graph.retrieve_sec_filings_node(create_initial_state(TICKER, ""))

    assert [chunk["text"] for chunk in update["filing_chunks"]] == DENSE_TEXTS
    assert "errors" not in update


@pytest.mark.parametrize("broken_reranker", BREAKS, indirect=True)
async def test_dense_fallback_chunks_say_they_were_not_reranked_and_why(
    broken_reranker: FakeFoundry, request: pytest.FixtureRequest
) -> None:
    """test-plan §25: App rerank fallback (each chunk is marked `reranked` = false with the
    failure reason in `rerank_fallback_reason`, never a credential)."""
    chunks = await graph._retrieve_filing_chunks(_store(), TICKER, COMPANY)

    assert [chunk.id for chunk in chunks] == DENSE_IDS
    for chunk in chunks:
        assert chunk.metadata.get("reranked") is False
        reason = chunk.metadata.get("rerank_fallback_reason")
        assert isinstance(reason, str) and reason.strip()
        assert "test-key" not in reason
        assert "reranker_score" not in chunk.metadata
    if request.node.callspec.params["broken_reranker"] in UNCONFIGURED:
        assert broken_reranker.requests == []
    else:
        assert broken_reranker.requests


async def test_a_reranker_failing_part_way_does_not_mix_reranked_and_dense_chunks(
    foundry_rerank: FakeFoundry,
) -> None:
    """test-plan §25: App rerank fallback (if any topic query's rerank fails, the step returns the
    dense results for all four queries): the first rerank request succeeds, the rest fail."""
    foundry_rerank.fail_after = 1
    foundry_rerank.fail_status = 503

    chunks = await graph._retrieve_filing_chunks(_store(), TICKER, COMPANY)

    assert [chunk.id for chunk in chunks] == DENSE_IDS
    assert all(chunk.metadata.get("reranked") is False for chunk in chunks)
    assert all(chunk.metadata.get("rerank_fallback_reason") for chunk in chunks)


class _FailingSearchStore(FakeRetrievalStore):
    """A store whose candidate search breaks with something that is not a reranker failure; its
    by-ticker search still serves the dense results."""

    def search(
        self, query: str, filters: SearchFilters | None = None, n_results: int = 5
    ) -> SearchResult:
        raise RuntimeError("private retrieval details")

    def search_by_ticker(
        self, query: str, ticker: str, n_results: int = 5, section: str | None = None
    ) -> SearchResult:
        return FakeRetrievalStore.search(self, query, SearchFilters(ticker=ticker), n_results)


async def test_a_store_failure_is_not_swallowed_into_the_dense_fallback(
    foundry_rerank: FakeFoundry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test-plan §25: App rerank fallback (other exceptions are not swallowed): a store error keeps
    the step's failure record instead of returning dense chunks."""
    _use(monkeypatch, _FailingSearchStore(POINTS))

    update = await graph.retrieve_sec_filings_node(create_initial_state(TICKER, COMPANY))

    assert update["filing_chunks"] == []
    assert [(e["code"], e["message"]) for e in update["errors"]] == [
        ("filing_retrieval_failed", f"Filing retrieval failed for {TICKER}.")
    ]
    assert "private" not in str(update)
    assert foundry_rerank.requests == []
