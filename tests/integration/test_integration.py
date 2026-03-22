from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as main
from api.run_store import FileBackedRunStore


class FakeVectorStore:
    def __init__(self) -> None:
        self.count = 12

    def get_stats(self) -> dict:
        return {
            "total_documents": self.count,
            "backend": "fake-chroma",
        }

    def search_by_ticker(self, query: str, ticker: str, n_results: int = 1):
        return SimpleNamespace(has_results=True, chunks=[{"ticker": ticker, "text": "fake chunk"}])

def _fake_agent_state(ticker: str, company_name: str | None = None) -> dict:
    company = company_name or f"{ticker} Corp"
    return {
        "ticker": ticker,
        "company_name": company,
        "executive_summary": f"{ticker} looks operationally healthy in the mocked pipeline.",
        "investment_memo": f"# Memo\n\nMocked memo for {ticker}.",
        "stock_data": {
            "ticker": ticker,
            "company_name": company,
            "current_price": 123.45,
            "price_change_percent": 1.23,
            "market_cap": 2_500_000_000_000,
            "pe_ratio": 31.2,
            "fifty_two_week_high": 200.0,
            "fifty_two_week_low": 100.0,
            "target_price": 210.0,
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
                "title": f"{ticker} launches something shiny",
                "url": f"https://example.com/{ticker.lower()}-news",
                "source": "Example News",
                "snippet": "Mocked article used for integration testing.",
                "published_date": "2026-03-16",
                "relevance_score": 0.91,
            }
        ],
        "citations": [
            {
                "index": 1,
                "source_type": "news",
                "title": f"{ticker} mocked citation",
                "url": f"https://example.com/{ticker.lower()}-citation",
                "date": "2026-03-16",
            }
        ],
        "errors": [],
        "started_at": "2026-03-16T00:00:00+00:00",
        "completed_at": "2026-03-16T00:00:01+00:00",
        "execution_time_ms": 1000.0,
    }

async def fake_run_agent(ticker: str, company_name: str | None = None):
    return _fake_agent_state(ticker=ticker, company_name=company_name)


async def fake_check_ollama_health() -> bool:
    return True


async def fake_check_langsmith_connection() -> dict:
    return {"connected": True}


async def fake_ingest_10k_for_ticker(ticker: str):
    return SimpleNamespace(
        success=True,
        total_chunks=7,
        sections_processed=["business", "risk_factors", "md&a"],
        filing_date="2025-11-01",
        error=None,
    )

def test_runtime_hardened_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    run_store_path = tmp_path / "run_store.json"
    temp_store = FileBackedRunStore(run_store_path)

    monkeypatch.setattr(main, "run_store", temp_store)
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    monkeypatch.setattr(main, "check_ollama_health", fake_check_ollama_health)
    monkeypatch.setattr(main, "check_langsmith_connection", fake_check_langsmith_connection)
    monkeypatch.setattr(main, "ingest_10k_for_ticker", fake_ingest_10k_for_ticker)

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        health_payload = health.json()
        assert health_payload["status"] == "healthy"
        assert health_payload["components"]["ollama"]["ok"] is True
        assert health_payload["components"]["vector_store"]["ok"] is True

        ingest = client.post(
            "/ingest",
            json={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "force_refresh": False,
            },
        )
        assert ingest.status_code == 200
        ingest_payload = ingest.json()
        assert ingest_payload["status"] == "success"
        assert ingest_payload["chunks_created"] == 7
        assert ingest_payload["sections_processed"] == ["business", "risk_factors", "md&a"]

        sync_analysis = client.post(
            "/analyze",
            json={
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "include_filing_analysis": True,
                "include_news_sentiment": True,
                "max_news_articles": 5,
            },
        )
        assert sync_analysis.status_code == 200
        sync_payload = sync_analysis.json()
        assert sync_payload["ticker"] == "AAPL"
        assert sync_payload["status"] == "completed"
        assert sync_payload["investment_memo"]
        assert len(sync_payload["citations"]) == 1

        async_analysis = client.post(
            "/analyze/async",
            json={
                "ticker": "MSFT",
                "company_name": "Microsoft",
                "include_filing_analysis": True,
                "include_news_sentiment": True,
                "max_news_articles": 5,
            },
        )
        assert async_analysis.status_code == 200
        async_payload = async_analysis.json()
        assert async_payload["ticker"] == "MSFT"
        assert async_payload["status"] in {"pending", "running", "completed"}

        job_id = async_payload["job_id"]
        final_job = None
        for _ in range(20):
            response = client.get(f"/jobs/{job_id}")
            assert response.status_code == 200
            final_job = response.json()
            if final_job["status"] == "completed":
                break
            time.sleep(0.05)

        assert final_job is not None
        assert final_job["status"] == "completed"
        assert final_job["ticker"] == "MSFT"

        stats = client.get("/stats")
        assert stats.status_code == 200
        stats_payload = stats.json()
        assert "vector_store" in stats_payload
        assert "run_store" in stats_payload
        assert stats_payload["run_store"]["completed"] >= 1
        assert stats_payload["run_store"]["path"].endswith("run_store.json")

        # Simulate process restart by swapping in a new run-store instance
        main.run_store = FileBackedRunStore(run_store_path)

        persisted_job = client.get(f"/jobs/{job_id}")
        assert persisted_job.status_code == 200
        persisted_payload = persisted_job.json()
        assert persisted_payload["job_id"] == job_id
        assert persisted_payload["status"] == "completed"