"""Trace: docs/test-plan.md §25, "Retrieval trace in the app run" (FR.0, 2026-11-24), the metrics
half; docs/sprint-plan.md "FR.0" (b).

A live filings step (cache miss) exposes on `/metrics` the histograms `retrieval_dense_score_gap`
and `retrieval_kept_dense_rank` with `stage="filings"`: the gap once per traced topic query, the
kept dense rank once per kept chunk of each traced query (four queries of K = 3: twelve). The
company-name lookup is not traced; a cache hit and a dense fallback observe neither. The registry
is process-wide, so every assertion is a delta of the `_count` and `_sum` series read from the
exposition text. The reranker is the loopback Foundry fake (word-overlap relevance), the store the
in-memory fake; the expected values are computed from them in tests/unit/trace_support.py.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import app.main as api
from app.agents import graph
from app.agents.state import MAX_FILING_CHUNKS, create_initial_state
from app.services.run_store import FileBackedRunStore
from tests.unit.fake_foundry import FakeFoundry, relevance
from tests.unit.fake_retrieval_store import FakeRetrievalStore
from tests.unit.trace_support import (
    FILING_TEMPLATES,
    GAP_HISTOGRAM,
    RANK_HISTOGRAM,
    expected_filing_traces,
    filing_points,
    histogram_totals,
)

TICKER = "AAPL"
COMPANY = "Acme"
AUTH = {"X-API-Key": "private-auth-value"}
QUERIES = [template.format(company=COMPANY) for template in FILING_TEMPLATES]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeRetrievalStore:
    store = FakeRetrievalStore(filing_points(TICKER, COMPANY) + filing_points("MSFT", "Beta"))
    monkeypatch.setattr(graph, "get_vector_store", lambda: store)
    return store


@pytest.fixture
def metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Reads the four totals `(gap count, gap sum, rank count, rank sum)` from `/metrics`."""
    monkeypatch.setattr(api, "run_store", FileBackedRunStore(tmp_path / "runs.json"))
    monkeypatch.setattr(api.settings, "api_key", "private-auth-value")
    monkeypatch.setattr(api.settings, "jwt_issuers", [])
    transport = httpx.ASGITransport(app=api.app)

    async def read() -> tuple[float, float, float, float]:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers=AUTH)
        assert response.status_code == 200
        return (
            *histogram_totals(response.text, GAP_HISTOGRAM),
            *histogram_totals(response.text, RANK_HISTOGRAM),
        )

    return read


def _delta(before: tuple[float, ...], after: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(b - a for a, b in zip(before, after, strict=True))


def _expected_totals(store: FakeRetrievalStore) -> tuple[float, float, float, float]:
    traces = expected_filing_traces(store, TICKER, COMPANY, relevance)
    gaps = [t["dense_score_gap"] for t in traces if t["dense_score_gap"] is not None]
    ranks = [rank for t in traces for rank in t["kept_dense_ranks"]]
    return float(len(gaps)), sum(gaps), float(len(ranks)), float(sum(ranks))


async def _step(company_name: str = COMPANY) -> dict[str, Any]:
    return await graph.retrieve_sec_filings_node(create_initial_state(TICKER, company_name))


def test_the_topic_query_templates_are_the_four_the_filings_step_sends() -> None:
    """test-plan §25: Retrieval trace in the app run (the four `{company}` templates, in query
    order, that the trace and the workload report are keyed on)."""
    assert getattr(graph, "FILING_TOPIC_QUERIES") == tuple(FILING_TEMPLATES)


@pytest.mark.parametrize("company_name", ["", COMPANY], ids=["name-from-index", "name-in-state"])
async def test_a_live_step_observes_the_gap_per_query_and_the_rank_per_kept_chunk(
    foundry_rerank: FakeFoundry,
    store: FakeRetrievalStore,
    metrics: Any,
    company_name: str,
) -> None:
    """test-plan §25: Retrieval trace in the app run (`retrieval_dense_score_gap` observed once
    per traced query, `retrieval_kept_dense_rank` once per kept chunk of each traced query, both
    `stage="filings"`; the company-name lookup is not traced): four gaps and 4 x 3 ranks, with
    the sums the store and the reranker give."""
    before = await metrics()

    update = await _step(company_name)

    after = await metrics()
    assert sorted(request.body["query"] for request in foundry_rerank.requests) == sorted(QUERIES)
    gap_count, gap_sum, rank_count, rank_sum = _delta(before, after)
    expected = _expected_totals(store)
    assert (gap_count, rank_count) == (4.0, 12.0) == expected[::2]
    assert gap_sum == pytest.approx(expected[1])
    assert rank_sum == pytest.approx(expected[3])
    assert update["filing_chunks"]


async def test_the_filing_chunks_and_errors_are_what_an_untraced_step_returns(
    foundry_rerank: FakeFoundry, store: FakeRetrievalStore
) -> None:
    """test-plan §25: Retrieval trace in the app run (the filing chunks and errors are the same as
    without tracing): the kept chunks are the best five of the merged per-query top threes by
    reranker score, highest first."""
    update = await _step()

    traces = expected_filing_traces(store, TICKER, COMPANY, relevance)
    best: dict[str, float] = {}
    for trace in traces:
        by_id = {c["chunk_id"]: c["rerank_score"] for c in trace["candidates"]}
        for chunk_id in trace["kept_ids"]:
            best[chunk_id] = max(by_id[chunk_id], best.get(chunk_id, 0.0))
    ordered = sorted(best.values(), reverse=True)
    assert len(ordered) > MAX_FILING_CHUNKS
    assert ordered[MAX_FILING_CHUNKS - 1] > ordered[MAX_FILING_CHUNKS], "precondition: no tie"
    expected = sorted(best, key=lambda chunk_id: -best[chunk_id])[:MAX_FILING_CHUNKS]
    texts = {point.id: point.text for point in store.points}
    assert [chunk["text"] for chunk in update["filing_chunks"]] == [texts[c] for c in expected]
    assert not update.get("errors")


async def test_a_cache_hit_observes_neither_histogram(
    foundry_rerank: FakeFoundry, store: FakeRetrievalStore, metrics: Any
) -> None:
    """test-plan §25: Retrieval trace in the app run (a cache hit records no trace and observes
    neither histogram): the first step observes, the second step of the same ticker and company
    adds nothing."""
    start = await metrics()
    first = await _step()
    after_first = await metrics()
    foundry_rerank.requests.clear()

    second = await _step()

    assert _delta(start, after_first) == pytest.approx(_expected_totals(store))
    assert [chunk["text"] for chunk in second["filing_chunks"]] == [
        chunk["text"] for chunk in first["filing_chunks"]
    ]
    assert foundry_rerank.requests == []
    assert _delta(after_first, await metrics()) == (0.0, 0.0, 0.0, 0.0)


async def test_a_dense_fallback_observes_neither_histogram(
    foundry_rerank: FakeFoundry, store: FakeRetrievalStore, metrics: Any
) -> None:
    """test-plan §25: Retrieval trace in the app run (a query that fell back to dense search has
    no trace): while the reranker is down the step observes nothing; once it is back, the next
    step does."""
    foundry_rerank.status = 503
    before = await metrics()
    fallback = await _step()
    after_fallback = await metrics()
    assert fallback["filing_chunks"]
    assert all(chunk["text"] for chunk in fallback["filing_chunks"])
    foundry_rerank.status = None

    await _step()

    assert _delta(before, after_fallback) == (0.0, 0.0, 0.0, 0.0)
    assert _delta(after_fallback, await metrics()) == pytest.approx(_expected_totals(store))
