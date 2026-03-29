# tests/integration/test_workflow_verification.py

from types import SimpleNamespace

import pytest

from agents import graph as graph_mod


class StubVectorStore:
    def search_by_ticker(self, query: str, ticker: str, n_results: int = 3):
        chunk = SimpleNamespace(
            text=(
                "Apple sells hardware, software, and services, "
                "and reports ongoing business and market risks."
            ),
            section="Business",
            metadata={"filing_type": "10-K"},
            filing_date="2025-09-28",
            relevance_score=0.95,
        )
        return SimpleNamespace(chunks=[chunk])


class StubLLM:
    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=(
                "## Executive Summary\n"
                "Apple remains financially solid with positive recent coverage [1][2].\n\n"
                "## Company Overview\n"
                "Apple operates consumer hardware and services [2].\n\n"
                "## Conclusion\n"
                "The available evidence supports a cautious positive view [1][2]."
            )
        )


async def fake_search_company_news(query: str, max_results: int = 5):
    return [
        SimpleNamespace(
            title="Apple launches updated product lineup",
            url="https://example.com/apple-launch",
            source="Example News",
            snippet="Recent coverage says Apple launched updated products and investor sentiment stayed constructive.",
            published_date="2026-03-15",
        )
    ]


async def fake_get_stock_data(ticker: str):
    return SimpleNamespace(
        ticker=ticker,
        company_name="Apple Inc.",
        current_price=210.0,
        price_change_percent=1.2,
        market_cap=3_100_000_000_000,
        pe_ratio=28.4,
        fifty_two_week_high=220.0,
        fifty_two_week_low=165.0,
        volume=55_000_000,
        dividend_yield=0.45,
        sector="Technology",
        industry="Consumer Electronics",
        recommendation="buy",
    )


def fake_analyze_sentiment_batch(texts):
    return [SimpleNamespace(label="positive") for _ in texts]


@pytest.mark.asyncio
async def test_run_agent_real_graph_populates_verification_result(monkeypatch):
    monkeypatch.setattr(graph_mod, "search_company_news", fake_search_company_news)
    monkeypatch.setattr(graph_mod, "get_stock_data", fake_get_stock_data)
    monkeypatch.setattr(graph_mod, "get_vector_store", lambda: StubVectorStore())
    monkeypatch.setattr(graph_mod, "analyze_sentiment_batch", fake_analyze_sentiment_batch)
    monkeypatch.setattr(graph_mod, "get_llm", lambda temperature=0.7: StubLLM())
    monkeypatch.setattr(graph_mod, "get_tracer", lambda: None)

    result = await graph_mod.run_agent("AAPL", "Apple Inc.")

    assert result["investment_memo"]
    assert result["citation_evidence"]
    assert "verification_result" in result

    verification = result["verification_result"]
    assert isinstance(verification, dict)
    assert "passed" in verification
    assert "claims" in verification
    assert "orphan_citations" in verification
    assert verification["orphan_citations"] == []

    assert result["current_step"] == graph_mod.AgentStep.COMPLETE.value
    assert result.get("completed_at")
    assert result.get("execution_time_ms") is not None
