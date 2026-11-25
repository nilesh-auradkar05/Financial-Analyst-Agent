"""Trace: docs/test-plan.md §25, "Retrieval trace in the app run" (FR.0, 2026-11-24), the LangSmith
and API half; docs/sprint-plan.md "FR.0" (a).

A live filings step (cache miss) records on the `retrieve_filings` run's metadata
`retrieval_traces`: one trace per reranked topic query, in query order, with no chunk text. The
company-name lookup is not traced, a cache hit and a dense fallback record no trace, and when
ingest-on-miss retries retrieval the run carries the retry's traces. `/metrics` observes the two
histograms once per traced query and once per kept chunk (K = 3 over four queries: twelve).

The run goes through `POST /analyze` with the offline LangSmith collector of
tests/test_request_tracing.py (its `traced` and `evidence` fixtures), then the in-memory store and
the loopback Foundry rerank fake replace the evidence fakes' store and the reranker. Expected
traces are computed from them (tests/unit/trace_support.py). No network beyond 127.0.0.1.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.agents import graph
from app.components.retrieval.ingestion import IngestionResult
from tests.test_request_tracing import evidence, offline, traced  # noqa: F401
from tests.unit.fake_foundry import FakeFoundry, relevance
from tests.unit.fake_retrieval_store import FakeRetrievalStore
from tests.unit.trace_support import (
    FILING_TEMPLATES,
    GAP_HISTOGRAM,
    RANK_HISTOGRAM,
    assert_trace_equals,
    expected_filing_traces,
    filing_points,
    histogram_totals,
)

AUTH = {"X-API-Key": "private-auth-value"}
COMPANY = {"AAPL": "Acme", "MSFT": "Beta", "NVDA": "Gamma"}
pytestmark = pytest.mark.usefixtures("traced", "evidence")


@pytest.fixture
def collector(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("traced")


@pytest.fixture
def store(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> FakeRetrievalStore:
    """AAPL and MSFT indexed; set after the `evidence` fixture so the filings step reads it."""
    request.getfixturevalue("evidence")
    store = FakeRetrievalStore(filing_points("AAPL", "Acme") + filing_points("MSFT", "Beta"))
    monkeypatch.setattr(graph, "get_vector_store", lambda: store)
    return store


def _analyze(client: TestClient, ticker: str) -> None:
    response = client.post("/analyze", json={"ticker": ticker}, headers=AUTH)
    assert response.status_code == 200


def _totals(client: TestClient) -> tuple[float, float, float, float]:
    response = client.get("/metrics", headers=AUTH)
    assert response.status_code == 200
    return (
        *histogram_totals(response.text, GAP_HISTOGRAM),
        *histogram_totals(response.text, RANK_HISTOGRAM),
    )


def _delta(before: tuple[float, ...], after: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(b - a for a, b in zip(before, after, strict=True))


def _expected_totals(traces: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    gaps = [t["dense_score_gap"] for t in traces if t["dense_score_gap"] is not None]
    ranks = [rank for t in traces for rank in t["kept_dense_ranks"]]
    return float(len(gaps)), sum(gaps), float(len(ranks)), float(sum(ranks))


def _filings_runs(collector: Any) -> list[dict[str, Any]]:
    """The filings step's retriever runs (the graph node of the same name is a chain run)."""
    return [
        run
        for run in collector.runs.values()
        if run["name"] == "retrieve_filings" and run["run_type"] == "retriever"
    ]


def _recorded(run: dict[str, Any]) -> list[dict[str, Any]]:
    """The traces a `retrieve_filings` run carries (none when the run records none)."""
    return list(run["extra"]["metadata"].get("retrieval_traces") or [])


def test_a_live_filings_step_records_one_trace_per_topic_query_in_query_order(
    collector: Any, foundry_rerank: FakeFoundry, store: FakeRetrievalStore
) -> None:
    """test-plan §25: Retrieval trace in the app run (`retrieval_traces` on the `retrieve_filings`
    run: one trace per reranked topic query, in query order; the company-name lookup is not
    traced), and `/metrics` observes the gap once per traced query and the rank once per kept
    chunk of each."""
    with TestClient(api.app) as client:
        before = _totals(client)
        _analyze(client, "AAPL")
        after = _totals(client)

    (run,) = _filings_runs(collector)
    recorded = _recorded(run)
    expected = expected_filing_traces(store, "AAPL", COMPANY["AAPL"], relevance)
    assert [t["query"] for t in recorded] == [
        template.format(company=COMPANY["AAPL"]) for template in FILING_TEMPLATES
    ]
    assert all(t["ticker"] == "AAPL" and t["query"] != "AAPL" for t in recorded)
    for got, want in zip(recorded, expected, strict=True):
        assert_trace_equals(got, want, filing_type=False)
        assert got["candidate_pool"] == 15 and got["candidate_source"] == "dense"
    assert _delta(before, after) == pytest.approx(_expected_totals(expected))
    assert _delta(before, after)[::2] == (4.0, 12.0)


def test_the_recorded_traces_hold_no_chunk_text(
    collector: Any, foundry_rerank: FakeFoundry, store: FakeRetrievalStore
) -> None:
    """test-plan §25: Retrieval trace in the app run (the traces carry no chunk text): neither the
    recorded traces nor anything else the run exports holds a passage of the filing."""
    with TestClient(api.app) as client:
        _analyze(client, "AAPL")

    (run,) = _filings_runs(collector)
    recorded = _recorded(run)
    assert len(recorded) == 4
    text = json.dumps(recorded, default=str)
    for point in store.points:
        if point.metadata["ticker"] == "AAPL":
            assert point.text not in text
    assert "test-key" not in str(collector.runs)


def test_a_cache_hit_records_no_trace_and_observes_neither_histogram(
    collector: Any, foundry_rerank: FakeFoundry, store: FakeRetrievalStore
) -> None:
    """test-plan §25: Retrieval trace in the app run (a cache hit records no trace and observes
    neither histogram): the first analysis of a ticker records four traces, the second adds none."""
    with TestClient(api.app) as client:
        _analyze(client, "AAPL")
        after_first = _totals(client)
        foundry_rerank.requests.clear()

        _analyze(client, "AAPL")

        after_second = _totals(client)
    first, second = _filings_runs(collector)
    assert len(_recorded(first)) == 4
    assert foundry_rerank.requests == []
    assert _recorded(second) == []
    assert _delta(after_first, after_second) == (0.0, 0.0, 0.0, 0.0)


def test_a_query_that_fell_back_to_dense_has_no_trace(
    collector: Any, foundry_rerank: FakeFoundry, store: FakeRetrievalStore
) -> None:
    """test-plan §25: Retrieval trace in the app run (a query that fell back to dense search has no
    trace): with the reranker up, one ticker records four traces; with it down, the next ticker's
    step records none and observes nothing."""
    with TestClient(api.app) as client:
        _analyze(client, "AAPL")
        after_live = _totals(client)
        foundry_rerank.status = 503

        _analyze(client, "MSFT")

        after_fallback = _totals(client)
    live, fallback = _filings_runs(collector)
    assert len(_recorded(live)) == 4
    assert _recorded(fallback) == []
    assert _delta(after_live, after_fallback) == (0.0, 0.0, 0.0, 0.0)


def test_ingest_on_miss_records_the_traces_of_the_retry(
    collector: Any,
    foundry_rerank: FakeFoundry,
    store: FakeRetrievalStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """test-plan §25: Retrieval trace in the app run (when ingest-on-miss retries retrieval, the
    run carries the retry's traces): NVDA is not indexed; ingestion adds it; the run holds four
    traces, each with candidates from the ingested filing only, and the histograms observe them."""

    async def ingest(ticker: str, **_kwargs: Any) -> IngestionResult:
        points = filing_points(ticker, COMPANY[ticker])
        store.points.extend(points)
        return IngestionResult(ticker=ticker, filing_type="10-K", total_chunks=len(points))

    monkeypatch.setattr(graph, "ingest_10k_for_ticker", ingest)
    with TestClient(api.app) as client:
        before = _totals(client)
        _analyze(client, "NVDA")
        after = _totals(client)

    assert len(foundry_rerank.requests) == 4
    assert all(COMPANY["NVDA"] in r.body["query"] for r in foundry_rerank.requests)
    (run,) = _filings_runs(collector)
    recorded = _recorded(run)
    expected = expected_filing_traces(store, "NVDA", COMPANY["NVDA"], relevance)
    assert len(recorded) == 4
    assert [t["query"] for t in recorded] == [
        template.format(company=COMPANY["NVDA"]) for template in FILING_TEMPLATES
    ]
    for got, want in zip(recorded, expected, strict=True):
        assert got["candidates"]
        assert {c["chunk_id"] for c in got["candidates"]} <= {
            point.id for point in store.points if point.metadata["ticker"] == "NVDA"
        }
        assert_trace_equals(got, want, filing_type=False)
    assert _delta(before, after) == pytest.approx(_expected_totals(expected))
