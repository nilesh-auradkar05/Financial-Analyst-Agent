"""Trace: docs/test-plan.md §13, "Filing result cache (FILINGS-LATENCY, 2026-11-23; early in-process
slice of SPEC §11.3 RetrievalCache)"; the filings step's candidate pool of 15 traces to
docs/adr/ADR-0011 "Amendment 2026-11-23 (FILINGS-LATENCY)" and docs/retrieval-benchmark.md §7d (no
test-plan row states it yet: §25 "App filing retrieval reranks" is unamended).

A second filings step for the same ticker, company name and retrieval configuration within 1 h
returns the same filing chunks, with the same fetch time, and makes no embedding, search or rerank
call; after 1 h, or after the ticker is ingested in this process (ingest-on-miss or `POST /ingest`),
the next step retrieves afresh; a different company name misses; an empty result or a dense-fallback
result is not cached; `/metrics` counts hits and misses as `cache_events_total{stage="filings",
outcome}`.

The vector store is an in-memory fake that counts the searches it serves; the reranker is the
loopback Foundry fake (tests/unit/fake_foundry.py), which counts the requests it receives and
computes its scores from them. The clock is the fake clock (tests/conftest.py `fake_clock`), which
moves the cache's TTL. Each test starts from the empty cache the autouse fixture installs
(tests/conftest.py `fresh_filing_cache`). No network beyond 127.0.0.1, dummy credentials.
"""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

import app.main as api
from app.agents import graph
from app.agents.state import create_initial_state
from app.components.retrieval import ingestion
from app.components.retrieval.ingestion import IngestionResult
from app.components.retrieval.vector_store import IndexDocument, SearchFilters, SearchResult
from app.services.run_store import FileBackedRunStore
from tests.unit.fake_foundry import FakeFoundry
from tests.unit.fake_retrieval_store import FakeRetrievalStore
from tests.unit.test_ingestion import StubVectorStore

TICKER = "AAPL"
OTHER_TICKER = "MSFT"
COMPANY = "Acme"
FETCHED_FIRST = "2026-11-20T10:00:00+00:00"
FETCHED_LATER = "2026-11-22T10:00:00+00:00"
TOPIC_WORDS = (
    "Acme business description overview risk factors challenges financial performance "
    "revenue competition market position"
)
TTL = 3600.0
AUTH = {"X-API-Key": "private-auth-value"}


def _point(chunk_id: str, text: str, ticker: str = TICKER, fetched_at: str = FETCHED_FIRST) -> Any:
    return IndexDocument(
        id=chunk_id,
        text=text,
        metadata={
            "ticker": ticker,
            "filing_type": "10-K",
            "section_key": "business",
            "company_name": COMPANY,
            "filing_date": "2026-01-01",
            "fetched_at": fetched_at,
        },
    )


def _points(ticker: str = TICKER, fetched_at: str = FETCHED_FIRST, suffix: str = "") -> list[Any]:
    """Dense order is the list order. Three plain passages come first, so a plain dense search
    returns them for every query; the passages that answer the topic queries come next, then thirty
    filler passages, and a passage of the same ticker that answers every topic query sits at dense
    rank 41, far outside a pool of 15."""
    plain = [f"{ticker} plain passage {n}{suffix}." for n in range(3)]
    topical = [f"Business description overview of segment {n}{suffix}." for n in range(4)] + [
        f"Risk factors in manufacturing{suffix}.",
        f"Financial performance in the fourth quarter{suffix}.",
        f"Competition from new entrants{suffix}.",
    ]
    filler = [f"Widgets are catalogued under note {n}{suffix}." for n in range(30)]
    deep = [f"{TOPIC_WORDS}{suffix}."]
    texts = plain + topical + filler + deep
    return [_point(f"{ticker}-{n:02d}", text, ticker, fetched_at) for n, text in enumerate(texts)]


DEEP_TEXT = f"{TOPIC_WORDS}."
FIRST_FIFTEEN_TEXTS = {point.text for point in _points()[:15]}


class CountingStore(FakeRetrievalStore):
    """The in-memory store, recording the `n_results` of every search it serves."""

    def __init__(self, points: list[Any]) -> None:
        super().__init__(points)
        self.searches: list[int] = []

    def search(
        self, query: str, filters: SearchFilters | None = None, n_results: int = 5
    ) -> SearchResult:
        self.searches.append(n_results)
        return super().search(query, filters, n_results)

    def search_hybrid(
        self, query: str, filters: SearchFilters | None = None, n_results: int = 5
    ) -> SearchResult:
        self.searches.append(n_results)
        return super().search_hybrid(query, filters, n_results)

    def replace_points(self, points: list[Any]) -> None:
        """The index changed under the process (a re-ingest by another process, say)."""
        self.points = list(points)


@pytest.fixture
def filing_results() -> ModuleType:
    """The cache module under test, with the empty cache `fresh_filing_cache` installed."""
    return importlib.import_module("app.cache.filing_results")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> CountingStore:
    store = CountingStore(_points() + _points(OTHER_TICKER))
    monkeypatch.setattr(graph, "get_vector_store", lambda: store)
    return store


async def _step(company_name: str = COMPANY, ticker: str = TICKER) -> dict[str, Any]:
    return await graph.retrieve_sec_filings_node(create_initial_state(ticker, company_name))


def _texts(update: dict[str, Any]) -> list[str]:
    return [chunk["text"] for chunk in update["filing_chunks"]]


async def _cache_one_step(store: CountingStore, foundry: FakeFoundry) -> dict[str, Any]:
    """The first filings step: it retrieves, and the call counters are zeroed after it."""
    first = await _step()
    assert first["filing_chunks"]
    assert len(foundry.requests) == 4
    store.searches.clear()
    foundry.requests.clear()
    return first


def _revised_index() -> list[Any]:
    return _points(fetched_at=FETCHED_LATER, suffix=" (revised)")


def test_the_cache_keeps_a_result_for_one_hour(filing_results: ModuleType) -> None:
    """test-plan §13: Filing result cache (within 1 h)."""
    assert filing_results.FILING_CACHE_TTL_SECONDS == TTL


async def test_a_second_step_for_the_same_request_is_served_without_any_retrieval_call(
    foundry_rerank: FakeFoundry, store: CountingStore
) -> None:
    """test-plan §13: Filing result cache (the same ticker, company name and configuration within
    1 h returns the same filing chunks, with the same fetch time, and makes no embedding, search or
    rerank call). The index is rewritten between the steps, so a result recomputed from it would
    differ."""
    first = await _cache_one_step(store, foundry_rerank)
    store.replace_points(_revised_index())

    second = await _step()

    assert second["filing_chunks"] == first["filing_chunks"]
    assert second["evidence_fetched_at"] == first["evidence_fetched_at"]
    assert first["evidence_fetched_at"]["filings"] == FETCHED_FIRST
    assert store.searches == []
    assert foundry_rerank.requests == []


@pytest.mark.parametrize(
    ("elapsed", "served_from_cache"),
    [(TTL - 100, True), (TTL + 100, False)],
    ids=["within", "after"],
)
async def test_a_result_older_than_one_hour_is_retrieved_afresh(
    foundry_rerank: FakeFoundry,
    store: CountingStore,
    fake_clock: Any,
    elapsed: float,
    served_from_cache: bool,
) -> None:
    """test-plan §13: Filing result cache (within 1 h from the cache; after 1 h the next step
    retrieves afresh, with the new index's chunks and fetch time)."""
    first = await _cache_one_step(store, foundry_rerank)
    store.replace_points(_revised_index())
    fake_clock.advance(elapsed)

    second = await _step()

    if served_from_cache:
        assert second["filing_chunks"] == first["filing_chunks"]
        assert store.searches == [] and foundry_rerank.requests == []
    else:
        assert all(text.endswith("(revised).") for text in _texts(second))
        assert second["evidence_fetched_at"]["filings"] == FETCHED_LATER
        assert store.searches and len(foundry_rerank.requests) == 4


@pytest.mark.parametrize(
    "company_name, ticker",
    [("Acme Holdings", TICKER), (COMPANY, OTHER_TICKER)],
    ids=["other-company-name", "other-ticker"],
)
async def test_a_different_company_name_or_ticker_misses(
    foundry_rerank: FakeFoundry,
    store: CountingStore,
    company_name: str,
    ticker: str,
) -> None:
    """test-plan §13: Filing result cache (a different company name misses; the ticker is part of
    the request too)."""
    await _cache_one_step(store, foundry_rerank)

    other = await _step(company_name, ticker)

    assert other["filing_chunks"]
    assert store.searches
    assert len(foundry_rerank.requests) == 4


async def test_invalidating_a_ticker_drops_only_that_tickers_result(
    foundry_rerank: FakeFoundry, store: CountingStore, filing_results: ModuleType
) -> None:
    """test-plan §13: Filing result cache (the ticker's entries are dropped when it is ingested;
    `FILING_CACHE.invalidate(ticker)` is the public hook)."""
    await _step()
    await _step(ticker=OTHER_TICKER)
    store.searches.clear()
    foundry_rerank.requests.clear()

    filing_results.FILING_CACHE.invalidate(TICKER)
    await _step(ticker=OTHER_TICKER)
    assert store.searches == [] and foundry_rerank.requests == []
    await _step(ticker=TICKER)

    assert store.searches
    assert len(foundry_rerank.requests) == 4


@pytest.fixture
def ingest_world(monkeypatch: pytest.MonkeyPatch) -> StubVectorStore:
    """The ingestion path with a faked SEC lookup and a stub index: `ingest_10k_for_ticker` fetches
    a filing for the ticker and writes it through the real `ingest_filing`."""
    index = StubVectorStore()

    async def latest(ticker: str) -> Any:
        filing_text = "The company is exposed to supply-chain and regulatory risk. " * 90
        return SimpleNamespace(
            metadata=SimpleNamespace(
                ticker=ticker,
                filing_type="10-K",
                filing_date="2026-09-28",
                company_name=COMPANY,
                accession_number="0000320193-26-000001",
                source_url="https://www.sec.gov/Archives/example",
            ),
            sections={"Item 1A. Risk Factors": SimpleNamespace(content=filing_text)},
        )

    monkeypatch.setattr(ingestion, "get_latest_10k", latest)
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: index)
    return index


@pytest.mark.parametrize("path", ["ingest_function", "post_ingest"])
async def test_a_successful_ingest_of_the_ticker_makes_the_next_step_retrieve_afresh(
    foundry_rerank: FakeFoundry,
    store: CountingStore,
    ingest_world: StubVectorStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    path: str,
) -> None:
    """test-plan §13: Filing result cache (after the ticker is ingested in this process -
    ingest-on-miss or `POST /ingest` - the next step retrieves afresh)."""
    first = await _cache_one_step(store, foundry_rerank)
    store.replace_points(_revised_index())

    if path == "ingest_function":
        result = await ingestion.ingest_10k_for_ticker(TICKER)
        assert result.success
    else:
        monkeypatch.setattr(api, "run_store", FileBackedRunStore(tmp_path / "runs.json"))
        monkeypatch.setattr(api.settings, "api_key", "private-auth-value")
        monkeypatch.setattr(api.settings, "jwt_issuers", [])
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            posted = await client.post("/ingest", json={"ticker": TICKER}, headers=AUTH)
        assert posted.status_code == 200
    assert ingest_world.documents

    second = await _step()

    assert _texts(second) != _texts(first)
    assert all(text.endswith("(revised).") for text in _texts(second))
    assert second["evidence_fetched_at"]["filings"] == FETCHED_LATER
    assert len(foundry_rerank.requests) == 4


@pytest.mark.parametrize("outcome", ["other-ticker", "failed-ingest"])
async def test_an_ingest_that_does_not_touch_the_ticker_keeps_its_result(
    foundry_rerank: FakeFoundry,
    store: CountingStore,
    ingest_world: StubVectorStore,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """test-plan §13: Filing result cache (only a successful ingest of the ticker drops its entry:
    another ticker's ingest, or an ingest that fails, leaves it)."""
    first = await _cache_one_step(store, foundry_rerank)
    if outcome == "other-ticker":
        assert (await ingestion.ingest_10k_for_ticker(OTHER_TICKER)).success
    else:

        def failing(filing: Any, sections: Any = None, replace_existing: bool = False) -> Any:
            return IngestionResult(
                ticker=filing.metadata.ticker, filing_type="10-K", error="10-K ingestion failed."
            )

        monkeypatch.setattr(ingestion, "ingest_filing", failing)
        assert not (await ingestion.ingest_10k_for_ticker(TICKER)).success

    second = await _step()

    assert second["filing_chunks"] == first["filing_chunks"]
    assert store.searches == [] and foundry_rerank.requests == []


async def test_an_empty_result_is_not_cached(
    foundry_rerank: FakeFoundry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test-plan §13: Filing result cache (an empty result is not cached): a ticker with no
    indexed filing and no filing to ingest returns nothing; once its filing is indexed the very next
    step returns it."""
    empty = CountingStore([])
    monkeypatch.setattr(graph, "get_vector_store", lambda: empty)

    async def no_10k(ticker: str, **_kwargs: Any) -> IngestionResult:
        return IngestionResult(
            ticker=ticker, filing_type="10-K", error="none", error_code="filing_not_found"
        )

    monkeypatch.setattr(graph, "ingest_10k_for_ticker", no_10k)
    nothing = await _step()
    assert nothing["filing_chunks"] == []
    empty.replace_points(_points())

    later = await _step()

    assert later["filing_chunks"]
    assert len(foundry_rerank.requests) == 4


async def test_a_dense_fallback_result_is_not_cached(
    foundry_rerank: FakeFoundry, store: CountingStore
) -> None:
    """test-plan §13: Filing result cache (a dense-fallback result is not cached): while the
    reranker is down the step returns the plain dense chunks; once it is back the very next step
    reranks again and returns the reranked chunks."""
    foundry_rerank.status = 503
    fallback = await _step()
    assert _texts(fallback) == [point.text for point in _points()[:3]]
    foundry_rerank.status = None
    foundry_rerank.requests.clear()

    recovered = await _step()

    assert len(foundry_rerank.requests) == 4
    assert _texts(recovered) != _texts(fallback)
    assert not set(_texts(fallback)) & set(_texts(recovered))


async def test_the_filings_step_asks_for_fifteen_candidates_per_query(
    foundry_rerank: FakeFoundry, store: CountingStore
) -> None:
    """FILINGS-LATENCY pool of 15 (ADR-0011 amendment 2026-11-23; retrieval-benchmark §7d; the test-plan
    row is missing): each of the four topic queries asks the store for 15 candidates and reranks at
    most 15 documents, so the same-ticker passage at dense rank 41 that answers every query is never
    a candidate and never kept."""
    update = await _step()

    assert len(foundry_rerank.requests) == 4
    assert {len(request.body["documents"]) for request in foundry_rerank.requests} == {15}
    assert max(store.searches) == 15
    assert store.searches.count(15) == 4
    assert set(_texts(update)) <= FIRST_FIFTEEN_TEXTS
    assert DEEP_TEXT not in _texts(update)
    assert update["filing_chunks"]


def _cache_events(text: str, outcome: str) -> float:
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == "cache_events_total"
        and sample.labels.get("stage") == "filings"
        and sample.labels.get("outcome") == outcome
    )


async def test_metrics_count_filing_cache_hits_and_misses(
    foundry_rerank: FakeFoundry,
    store: CountingStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """test-plan §13: Filing result cache (`/metrics` counts hits and misses as
    `cache_events_total{stage="filings", outcome}`): a first step is one miss, a second one hit."""
    monkeypatch.setattr(api, "run_store", FileBackedRunStore(tmp_path / "runs.json"))
    monkeypatch.setattr(api.settings, "api_key", "private-auth-value")
    monkeypatch.setattr(api.settings, "jwt_issuers", [])
    transport = httpx.ASGITransport(app=api.app)

    async def counts(client: httpx.AsyncClient) -> tuple[float, float]:
        response = await client.get("/metrics", headers=AUTH)
        assert response.status_code == 200
        exposition = response.text
        return _cache_events(exposition, "hit"), _cache_events(exposition, "miss")

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hits, misses = await counts(client)
        await _step()
        assert await counts(client) == (hits, misses + 1)
        await _step()
        assert await counts(client) == (hits + 1, misses + 1)


async def test_the_step_is_not_cached_across_tests(
    foundry_rerank: FakeFoundry, store: CountingStore
) -> None:
    """Guards the autouse isolation fixture (tests/conftest.py): a step for the ticker retrieves
    even though earlier tests ran the same step, because each test starts with an empty cache."""
    await _step()

    assert store.searches
    assert len(foundry_rerank.requests) == 4


async def test_invalidating_a_ticker_that_was_never_cached_is_harmless(
    foundry_rerank: FakeFoundry, store: CountingStore, filing_results: ModuleType
) -> None:
    """test-plan §13: Filing result cache (a fresh `FilingResultCache()` takes no arguments;
    ingesting a ticker nobody asked about drops nothing and raises nothing)."""
    cache = filing_results.FilingResultCache()
    cache.invalidate("NVDA")
    await _cache_one_step(store, foundry_rerank)

    filing_results.FILING_CACHE.invalidate("NVDA")
    second = await _step()

    assert second["filing_chunks"]
    assert store.searches == [] and foundry_rerank.requests == []
