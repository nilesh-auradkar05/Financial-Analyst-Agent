from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dataops.contracts import SourceType
from dataops.evidence_release import load_snapshot_records
from dataops.snapshot_writer import EvidenceSnapshotWriter
from evaluation import quality_baseline


class LiveAgentShouldNotRun:
    async def ainvoke(self, state, config=None):
        raise AssertionError("live graph should not run when evidence records are provided")


def _write_snapshot(
    writer: EvidenceSnapshotWriter,
    *,
    source_type: SourceType,
    natural_key: str,
    payload: dict,
) -> None:
    writer.write_json(
        source_type=source_type,
        ticker="AAPL",
        natural_key=natural_key,
        payload=payload,
        fetcher_name=f"{source_type}_fetcher",
        fetcher_version="1",
        fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_evaluate_single_run_replays_frozen_evidence_without_live_fetch(monkeypatch, tmp_path) -> None:
    writer = EvidenceSnapshotWriter(tmp_path)
    _write_snapshot(
        writer,
        source_type="news_article",
        natural_key="https://news.example.com/aapl",
        payload={
            "title": "Apple services revenue rises",
            "url": "https://news.example.com/aapl",
            "source": "Example News",
            "snippet": "Apple services revenue rose as subscriptions expanded.",
            "published_date": "2026-07-03",
        },
    )
    _write_snapshot(
        writer,
        source_type="market_quote",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 220.5,
            "market_cap": 3_400_000_000_000,
        },
    )
    _write_snapshot(
        writer,
        source_type="sec_filing",
        natural_key="0000320193-26-000001",
        payload={
            "ticker": "AAPL",
            "accession_number": "0000320193-26-000001",
            "filing_type": "10-K",
            "filing_date": "2026-10-30",
            "sections": {
                "Business": {
                    "name": "Business",
                    "item": "Item 1",
                    "content": "Apple sells devices and services to customers worldwide.",
                }
            },
        },
    )
    _write_snapshot(
        writer,
        source_type="sentiment_score",
        natural_key="fake-model:aapl",
        payload={
            "text": "Apple services revenue rose as subscriptions expanded.",
            "label": "positive",
            "confidence": 0.91,
            "scores": {"positive": 0.91, "negative": 0.04, "neutral": 0.05},
            "model_name": "fake-model",
        },
    )
    records = load_snapshot_records(tmp_path)

    async def fake_draft_memo_node(state):
        assert state["news_articles"][0]["title"] == "Apple services revenue rises"
        assert state["stock_data"]["current_price"] == 220.5
        assert state["filing_chunks"][0]["section"] == "Business"
        assert state["sentiment_result"] == {
            "overall_sentiment": "positive",
            "positive_count": 1,
            "negative_count": 0,
            "neutral_count": 0,
        }
        return {
            "investment_memo": "Apple services revenue rose as subscriptions expanded [1].",
            "citation_evidence": [
                {
                    "index": 1,
                    "source_type": "news",
                    "title": "Apple services revenue rises",
                    "text": "Apple services revenue rose as subscriptions expanded.",
                    "url": "https://news.example.com/aapl",
                    "date": "2026-07-03",
                }
            ],
            "errors": [],
        }

    async def fake_verify_memo_node(state):
        return {"verification_result": {"passed": True}, "errors": state.get("errors", [])}

    monkeypatch.setattr(quality_baseline, "draft_memo_node", fake_draft_memo_node)
    monkeypatch.setattr(quality_baseline, "verify_memo_node", fake_verify_memo_node)

    result = await quality_baseline.evaluate_single_run(
        LiveAgentShouldNotRun(),
        "AAPL",
        include_filing_analysis=True,
        include_news_sentiment=True,
        max_news_articles=10,
        checker="token",
        min_similarity=0.45,
        evidence_records_by_ticker={"AAPL": records},
    )

    assert result["ticker"] == "AAPL"
    assert result["evidence_count"] == 1
    assert result["news_articles"] == 1
    assert result["filing_chunks"] == 1
    assert result["grounded_claim_rate"] == 1.0
