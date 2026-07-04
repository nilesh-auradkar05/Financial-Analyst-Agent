from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dataops.live_capture import LiveEvidenceTools, record_evidence_release
from dataops.registry import ReleaseRegistry
from dataops.snapshot_writer import EvidenceSnapshotWriter


@pytest.mark.asyncio
async def test_record_evidence_release_captures_four_sources_and_registers_release(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    calls: list[tuple[str, str]] = []

    def _writer() -> EvidenceSnapshotWriter:
        return EvidenceSnapshotWriter(snapshot_root)

    def sec_extractor(ticker: str, **kwargs) -> object:
        calls.append(("sec", ticker))
        assert kwargs["record_evidence"] is True
        assert kwargs["evidence_output_dir"] == snapshot_root
        _writer().write_json(
            source_type="sec_filing",
            ticker=ticker,
            natural_key=f"{ticker}-10k",
            payload={"ticker": ticker, "sections": {"Business": {"content": "Business text"}}},
            fetcher_name="fake_sec",
            fetcher_version="1",
            fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        )
        return object()

    async def stock_fetcher(ticker: str, **kwargs) -> object:
        calls.append(("stock", ticker))
        assert kwargs["record_evidence"] is True
        assert kwargs["evidence_output_dir"] == snapshot_root
        _writer().write_json(
            source_type="market_quote",
            ticker=ticker,
            natural_key=f"{ticker}:2026-07-04T00:00:00+00:00",
            payload={"ticker": ticker, "current_price": 220.5},
            fetcher_name="fake_stock",
            fetcher_version="1",
            fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        )
        return object()

    async def news_search(query: str, **kwargs) -> list[object]:
        ticker = kwargs["evidence_ticker"]
        calls.append(("news", ticker))
        assert query == f"{ticker} stock news"
        assert kwargs["record_evidence"] is True
        assert kwargs["evidence_output_dir"] == snapshot_root
        _writer().write_json(
            source_type="news_article",
            ticker=ticker,
            natural_key=f"https://news.example.com/{ticker.lower()}",
            payload={"title": f"{ticker} news", "snippet": "Services revenue rose."},
            fetcher_name="fake_news",
            fetcher_version="1",
            fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        )
        return [SimpleNamespace(snippet="Services revenue rose.")]

    def sentiment_batch(texts: list[str], **kwargs) -> list[object]:
        ticker = kwargs["evidence_ticker"]
        calls.append(("sentiment", ticker))
        assert texts == ["Services revenue rose."]
        assert kwargs["record_evidence"] is True
        assert kwargs["evidence_output_dir"] == snapshot_root
        _writer().write_json(
            source_type="sentiment_score",
            ticker=ticker,
            natural_key=f"fake-model:{ticker}",
            payload={"text": texts[0], "label": "positive", "confidence": 0.9},
            fetcher_name="fake_sentiment",
            fetcher_version="1",
            fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        )
        return []

    manifest = await record_evidence_release(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        tickers=["aapl"],
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tools=LiveEvidenceTools(
            sec_extractor=sec_extractor,
            stock_fetcher=stock_fetcher,
            news_search=news_search,
            sentiment_batch=sentiment_batch,
        ),
    )

    registry = ReleaseRegistry(registry_root)

    assert manifest.release_id == "alpha-evidence:0.1.0"
    assert registry.read_active("evidence") == manifest
    assert calls == [
        ("sec", "AAPL"),
        ("stock", "AAPL"),
        ("news", "AAPL"),
        ("sentiment", "AAPL"),
    ]
