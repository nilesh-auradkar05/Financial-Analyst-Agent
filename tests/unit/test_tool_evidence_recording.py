from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import sentiment
from app.services.sentiment import SentimentResult
from app.services.tools import edgartools_sec_extractor as sec_tool
from app.services.tools import stock_data_tool, web_search_tool


def _snapshot_records(root: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("*.json")]


@pytest.mark.asyncio
async def test_web_search_records_news_snapshots_only_when_enabled(monkeypatch, tmp_path) -> None:
    class FakeTavilyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def search(self, **kwargs):
            return {
                "results": [
                    {
                        "title": "Apple services revenue rises",
                        "url": "https://news.example.com/aapl-services",
                        "content": "Apple services revenue rose as subscriptions expanded across its installed base.",
                        "score": 0.9,
                        "published_date": "2026-07-03",
                    }
                ]
            }

    monkeypatch.setattr(web_search_tool, "AsyncTavilyClient", FakeTavilyClient)
    monkeypatch.setattr(web_search_tool.settings.tavily, "api_key", "test-key")

    articles = await web_search_tool.search_company_news(
        "Apple AAPL stock news",
        max_results=1,
        evidence_output_dir=tmp_path,
    )

    assert len(articles) == 1
    assert _snapshot_records(tmp_path) == []

    recorded_articles = await web_search_tool.search_company_news(
        "Apple AAPL stock news",
        max_results=1,
        record_evidence=True,
        evidence_ticker="AAPL",
        evidence_output_dir=tmp_path,
    )

    records = _snapshot_records(tmp_path)
    assert [article.title for article in recorded_articles] == [articles[0].title]
    assert len(records) == 1
    assert records[0]["snapshot"]["source_type"] == "news_article"
    assert records[0]["snapshot"]["natural_key"] == "https://news.example.com/aapl-services"
    assert records[0]["payload"]["title"] == "Apple services revenue rises"


@pytest.mark.asyncio
async def test_stock_tool_records_market_quote_snapshot_only_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stock_data_tool, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(stock_data_tool, "yf", object())
    monkeypatch.setattr(stock_data_tool, "_with_retry", lambda func: func)
    monkeypatch.setattr(
        stock_data_tool,
        "_yfinance_fetch_sync",
        lambda ticker: {
            "longName": "Apple Inc.",
            "currentPrice": 220.5,
            "marketCap": 3_400_000_000_000,
            "sector": "Technology",
        },
    )

    quote = await stock_data_tool.get_stock_data("aapl", evidence_output_dir=tmp_path)

    assert quote.current_price == 220.5
    assert _snapshot_records(tmp_path) == []

    await stock_data_tool.get_stock_data(
        "aapl",
        record_evidence=True,
        evidence_output_dir=tmp_path,
    )

    records = _snapshot_records(tmp_path)
    assert len(records) == 1
    assert records[0]["snapshot"]["source_type"] == "market_quote"
    assert records[0]["snapshot"]["ticker"] == "AAPL"
    assert records[0]["payload"]["current_price"] == 220.5


def test_sec_extractor_records_filing_snapshot_only_when_enabled(monkeypatch, tmp_path) -> None:
    class FakeCompany:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def get_filings(self, form: str):
            return SimpleNamespace(latest=lambda: fake_filing)

    fake_filing = SimpleNamespace(
        cik="320193",
        accession_no="0000320193-26-000001",
        primary_document="aapl-20260926.htm",
        company="Apple Inc.",
        filing_date="2026-10-30",
        period_of_report="2026-09-26",
        obj=lambda: object(),
    )

    monkeypatch.setattr(sec_tool, "_configure_identity", lambda identity: FakeCompany)
    monkeypatch.setattr(
        sec_tool,
        "_extract_section_from_tenk",
        lambda **kwargs: "Apple section content. " * 100,
    )

    payload = sec_tool.extract_latest_10k_with_edgartools(
        "aapl",
        evidence_output_dir=tmp_path,
    )

    assert payload.accession_number == "0000320193-26-000001"
    assert _snapshot_records(tmp_path) == []

    sec_tool.extract_latest_10k_with_edgartools(
        "aapl",
        record_evidence=True,
        evidence_output_dir=tmp_path,
    )

    records = _snapshot_records(tmp_path)
    assert len(records) == 1
    assert records[0]["snapshot"]["source_type"] == "sec_filing"
    assert records[0]["snapshot"]["natural_key"] == "0000320193-26-000001"
    assert records[0]["payload"]["company_name"] == "Apple Inc."


def test_sentiment_records_score_snapshots_only_when_enabled(monkeypatch, tmp_path) -> None:
    class FakeAnalyzer:
        def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
            return [
                SentimentResult(
                    text=text,
                    label="positive",
                    confidence=0.91,
                    scores={"positive": 0.91, "negative": 0.04, "neutral": 0.05},
                )
                for text in texts
            ]

    monkeypatch.setattr(sentiment, "_get_default_analyzer", lambda: FakeAnalyzer())

    results = sentiment.analyze_sentiment_batch(
        ["Apple services revenue rose."],
        evidence_output_dir=tmp_path,
    )

    assert results[0].label == "positive"
    assert _snapshot_records(tmp_path) == []

    sentiment.analyze_sentiment_batch(
        ["Apple services revenue rose."],
        record_evidence=True,
        evidence_ticker="AAPL",
        evidence_output_dir=tmp_path,
    )

    records = _snapshot_records(tmp_path)
    assert len(records) == 1
    assert records[0]["snapshot"]["source_type"] == "sentiment_score"
    assert records[0]["snapshot"]["ticker"] == "AAPL"
    assert records[0]["payload"]["label"] == "positive"
