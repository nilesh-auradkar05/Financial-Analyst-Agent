from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.run_store import FileBackedRunStore


class StubSearchResult:
    def __init__(self, has_results: bool = True, chunks: list[dict[str, Any]] | None = None):
        self.has_results = has_results
        self.chunks = chunks or [{"chunk_id": "chunk-1"}]


class StubVectorStore:
    def __init__(self) -> None:
        self.count = 7

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "stub",
            "document_count": self.count,
            "tickers": ["AAPL", "MSFT"],
        }

    def search_by_ticker(self, query: str, ticker: str, n_results: int = 1) -> StubSearchResult:
        return StubSearchResult(
            has_results=True,
            chunks=[
                {
                    "chunk_id": f"{ticker}-chunk-1",
                    "query": query,
                    "ticker": ticker,
                }
            ],
        )


class StubIngestResult:
    success = True
    total_chunks = 8
    sections_processed = ["business", "risk_factors", "md&a"]
    filing_date = "2025-09-28"
    error = None


async def fake_check_ollama_health() -> bool:
    return True


async def fake_check_langsmith_connection() -> dict[str, bool]:
    return {"connected": True}


async def fake_ingest_10k_for_ticker(ticker: str) -> StubIngestResult:
    return StubIngestResult()


async def fake_run_agent(ticker: str, company_name: str | None):
    company = company_name or {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
    }.get(ticker, f"{ticker} Corp.")

    return {
        "ticker": ticker,
        "company_name": company,
        "executive_summary": f"{company} remains financially solid.",
        "investment_memo": f"# Investment Memo\n\nBull case for {ticker}.",
        "stock_data": {
            "ticker": ticker,
            "company_name": company,
            "current_price": 189.12,
            "price_change_percent": 1.42,
            "market_cap": 3_000_000_000_000,
            "pe_ratio": 31.8,
            "fifty_two_week_high": 199.62,
            "fifty_two_week_low": 164.08,
            "target_price": 205.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        "sentiment_result": {
            "overall_sentiment": "positive",
            "positive_count": 4,
            "negative_count": 1,
            "neutral_count": 0,
            "average_positive_score": 0.84,
            "average_negative_score": 0.22,
        },
        "news_articles": [
            {
                "title": f"{company} launches new product line",
                "url": f"https://example.com/{ticker.lower()}-launch",
                "source": "Example News",
                "snippet": "Launch expands addressable market.",
                "published_date": "2026-03-15",
                "relevance_score": 0.93,
            }
        ],
        "citations": [
            {
                "index": 1,
                "source_type": "news",
                "title": f"{company} launches new product line",
                "url": f"https://example.com/{ticker.lower()}-launch",
                "date": "2026-03-15",
            }
        ],
        "errors": [],
        "started_at": "2026-03-15T12:00:00+00:00",
        "completed_at": "2026-03-15T12:00:02+00:00",
        "execution_time_ms": 2000,
    }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_store = FileBackedRunStore(tmp_path / "run_store.json")

    monkeypatch.setattr(api_main, "run_store", temp_store)
    monkeypatch.setattr(api_main, "get_vector_store", lambda: StubVectorStore())
    monkeypatch.setattr(api_main, "check_ollama_health", fake_check_ollama_health)
    monkeypatch.setattr(api_main, "check_langsmith_connection", fake_check_langsmith_connection)
    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", fake_ingest_10k_for_ticker)
    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_endpoint_returns_component_statuses(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["components"]["ollama"]["ok"] is True
    assert payload["components"]["vector_store"]["ok"] is True
    assert payload["components"]["langsmith"]["connected"] is True


def test_ingest_and_ingest_status_endpoints(client: TestClient):
    ingest_response = client.post(
        "/ingest",
        json={"ticker": "AAPL", "filing_type": "10-K"},
    )

    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert ingest_payload["ticker"] == "AAPL"
    assert ingest_payload["status"] == "success"
    assert ingest_payload["chunks_created"] == 8
    assert ingest_payload["sections_processed"] == ["business", "risk_factors", "md&a"]

    status_response = client.get("/ingest/AAPL")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload == {
        "ticker": "AAPL",
        "indexed": True,
        "document_count": 1,
    }


def test_sync_analysis_endpoint_returns_full_analysis_payload(client: TestClient):
    response = client.post(
        "/analyze",
        json={"ticker": "AAPL", "company_name": "Apple Inc."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["company_name"] == "Apple Inc."
    assert payload["status"] == "completed"
    assert payload["executive_summary"] == "Apple Inc. remains financially solid."
    assert payload["investment_memo"].startswith("# Investment Memo")
    assert payload["stock_data"]["market_cap_formatted"] == "$3.00T"
    assert payload["sentiment"]["overall_sentiment"] == "positive"
    assert len(payload["news_articles"]) == 1
    assert len(payload["citations"]) == 1
    assert payload["errors"] == []


def test_async_analysis_creates_job_and_persists_completed_result(client: TestClient):
    create_response = client.post(
        "/analyze/async",
        json={"ticker": "MSFT", "company_name": "Microsoft Corp."},
    )

    assert create_response.status_code == 200
    create_payload = create_response.json()
    assert create_payload["ticker"] == "MSFT"
    assert create_payload["status"] in {"pending", "running", "completed"}

    job_id = create_payload["job_id"]
    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["job_id"] == job_id
    assert status_payload["ticker"] == "MSFT"
    assert status_payload["status"] == "completed"

    record = api_main.run_store.get_run(job_id)
    assert record is not None
    assert record.status == "completed"
    assert record.result is not None
    assert record.result["ticker"] == "MSFT"
    assert record.result["investment_memo"].startswith("# Investment Memo")


def test_stats_endpoint_reports_vector_and_run_store_counts(client: TestClient):
    response = client.post(
        "/analyze/async",
        json={"ticker": "AAPL", "company_name": "Apple Inc."},
    )
    assert response.status_code == 200

    stats_response = client.get("/stats")
    assert stats_response.status_code == 200
    payload = stats_response.json()

    assert payload["vector_store"] == {
        "backend": "stub",
        "document_count": 7,
        "tickers": ["AAPL", "MSFT"],
    }
    assert payload["run_store"]["total_runs"] == 1
    assert payload["run_store"]["completed"] == 1
    assert payload["run_store"]["failed"] == 0