"""Trace: docs/test-plan.md §16, analysis without filings and retrieval failure."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from app.agents import graph
from app.agents.state import apply_update, create_initial_state
from app.components.retrieval import ingestion
from app.components.retrieval.ingestion import IngestionResult


class FakeStore:
    """In-memory store: search returns what was written for the ticker, best first."""

    def __init__(self, docs: dict[str, list[str]] | None = None, searchable: bool = True):
        self.docs = docs or {}
        self.searchable = searchable

    def count_documents(self, filters=None):
        return len(self.docs.get(filters.ticker, []))

    def search_by_ticker(self, query, ticker, n_results=3, section=None):
        texts = self.docs.get(ticker, []) if self.searchable else []
        return SimpleNamespace(chunks=[
            SimpleNamespace(id=f"{ticker}-{n}", text=text, section="Business",
                            metadata={"filing_type": "10-K", "company_name": f"{ticker} Inc."},
                            filing_date="2026-01-01", relevance_score=1 / (n + 1))
            for n, text in enumerate(texts[:n_results])
        ])


def use_store(monkeypatch, store: FakeStore) -> FakeStore:
    monkeypatch.setattr(graph, "get_vector_store", lambda: store)
    return store


def forbid_ingestion(monkeypatch) -> None:
    async def ingest(ticker, **_kwargs):
        raise AssertionError(f"ingestion triggered for {ticker}")

    monkeypatch.setattr(graph, "ingest_10k_for_ticker", ingest)


async def no_10k(ticker, **_kwargs):
    return IngestionResult(ticker=ticker, filing_type="10-K", error="none", error_code="filing_not_found")


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval_fails", [False, True])
async def test_missing_filing_keeps_available_evidence_and_verifies(monkeypatch, retrieval_fails):
    state = create_initial_state("AAPL", "Apple Inc.")
    state["stock_data"] = {"ticker": "AAPL", "current_price": 123.0}
    state["news_articles"] = [{"title": "Market update", "snippet": "Shares rose.", "source": "Wire"}]
    state["sentiment_result"] = {"overall_sentiment": "positive", "positive_count": 1}

    use_store(monkeypatch, FakeStore())
    monkeypatch.setattr(graph, "ingest_10k_for_ticker", no_10k)
    if retrieval_fails:
        def search(*_args):
            raise RuntimeError("secret retrieval details")
        monkeypatch.setattr(graph, "_search_filing_chunks", search)

    state = apply_update(state, await graph.retrieve_sec_filings_node(state))

    class MemoLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="# Investment Memo\n\nMarket evidence is limited.")

    monkeypatch.setattr(graph, "get_llm", lambda _settings: MemoLLM())
    state = apply_update(state, await graph.draft_memo_node(state))
    verification = await graph.verify_memo_node(state)

    assert state["stock_data"]["current_price"] == 123.0
    assert len(state["news_articles"]) == 1
    assert state["sentiment_result"]["overall_sentiment"] == "positive"
    assert state.get("filing_chunks") == []
    assert "SEC Filings: Not Available" in state.get("investment_memo", "")
    assert "AAPL" in state.get("investment_memo", "")
    assert all(item["source_type"] != "sec_filing" for item in state.get("citations", []))
    assert "verification_result" in verification
    reason = next(error["message"] for error in state.get("errors", []) if error["step"] == "retrieve_filings")
    assert reason == ("Filing retrieval failed for AAPL." if retrieval_fails else "No indexed filings found for AAPL.")


@pytest.mark.asyncio
async def test_disabled_filing_analysis_has_no_missing_warning(monkeypatch):
    state = create_initial_state("AAPL", include_filing_analysis=False)
    retrieval = await graph.retrieve_sec_filings_node(state)
    state.update(retrieval)

    class MemoLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="# Memo\n\nNo conclusion.")

    monkeypatch.setattr(graph, "get_llm", lambda _settings: MemoLLM())
    draft = await graph.draft_memo_node(state)
    assert "SEC Filings: Not Available" not in draft["investment_memo"]
    assert not any(error["step"] == "retrieve_filings" for error in draft["errors"])


# Trace: docs/test-plan.md §16, ingest on miss (2026-10-10).

async def test_unindexed_ticker_is_ingested_then_retrieved(monkeypatch):
    store = use_store(monkeypatch, FakeStore())

    async def ingest(ticker, **_kwargs):
        store.docs[ticker] = [f"{ticker} 10-K passage {n}" for n in range(3)]
        return IngestionResult(ticker=ticker, filing_type="10-K", total_chunks=3)

    monkeypatch.setattr(graph, "ingest_10k_for_ticker", ingest)

    update = await graph.retrieve_sec_filings_node(create_initial_state("NET"))

    assert [chunk["text"] for chunk in update["filing_chunks"]] == [f"NET 10-K passage {n}" for n in range(3)]
    assert "errors" not in update


async def test_ticker_without_a_10k_reports_filing_not_found(monkeypatch):
    use_store(monkeypatch, FakeStore())
    monkeypatch.setattr(graph, "ingest_10k_for_ticker", no_10k)

    update = await graph.retrieve_sec_filings_node(create_initial_state("GSPC"))

    assert update["filing_chunks"] == []
    assert [(e["code"], e["message"]) for e in update["errors"]] == [
        ("filing_not_found", "No indexed filings found for GSPC."),
    ]


@pytest.mark.parametrize("failure", ["error_result", "raises", "exceeds_bound"])
async def test_failed_ingestion_is_distinct_from_no_filings(monkeypatch, failure):
    use_store(monkeypatch, FakeStore())
    monkeypatch.setattr(graph, "INGEST_TIMEOUT_SECONDS", 0.05)

    async def ingest(ticker, **_kwargs):
        if failure == "raises":
            raise RuntimeError("private upstream details")
        if failure == "exceeds_bound":
            await asyncio.sleep(1)
        return IngestionResult(ticker=ticker, filing_type="10-K", error="10-K ingestion failed.")

    monkeypatch.setattr(graph, "ingest_10k_for_ticker", ingest)

    update = await graph.retrieve_sec_filings_node(create_initial_state("AAPL"))

    assert update["filing_chunks"] == []
    assert [(e["code"], e["message"]) for e in update["errors"]] == [
        ("filing_ingestion_failed", "Filing ingestion failed for AAPL."),
    ]


@pytest.mark.parametrize("searchable", [True, False])
async def test_indexed_ticker_is_never_reingested(monkeypatch, searchable):
    use_store(monkeypatch, FakeStore({"AAPL": ["AAPL 10-K passage"]}, searchable=searchable))
    forbid_ingestion(monkeypatch)

    update = await graph.retrieve_sec_filings_node(create_initial_state("AAPL"))

    assert [chunk["text"] for chunk in update["filing_chunks"]] == (["AAPL 10-K passage"] if searchable else [])
    assert [e["code"] for e in update.get("errors", [])] == ([] if searchable else ["filing_not_found"])


async def test_disabled_filing_analysis_never_ingests(monkeypatch):
    use_store(monkeypatch, FakeStore())
    forbid_ingestion(monkeypatch)

    update = await graph.retrieve_sec_filings_node(create_initial_state("AAPL", include_filing_analysis=False))

    assert update["filing_chunks"] == []
    assert "errors" not in update


async def test_ingestion_work_stays_off_the_event_loop(monkeypatch):
    async def latest(ticker):
        return SimpleNamespace(ticker=ticker)

    def embed_and_write(filing, sections=None, replace_existing=False):
        time.sleep(0.4)  # embedding + store writes hold their thread
        return IngestionResult(ticker=filing.ticker, filing_type="10-K", total_chunks=1)

    monkeypatch.setattr(ingestion, "get_latest_10k", latest)
    monkeypatch.setattr(ingestion, "ingest_filing", embed_and_write)

    running = asyncio.create_task(ingestion.ingest_10k_for_ticker("AAPL"))
    started = time.perf_counter()
    await asyncio.sleep(0.05)
    lag = time.perf_counter() - started - 0.05
    result = await running

    assert result.total_chunks == 1
    assert lag < 0.1, f"event loop stalled {lag:.2f}s behind ingestion"
