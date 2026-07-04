from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dataops.contracts import DatasetReleaseManifest, EvidenceSnapshot


def test_evidence_snapshot_id_is_deterministic_and_payload_sensitive() -> None:
    fetched_at = datetime(2026, 7, 4, tzinfo=timezone.utc)

    first = EvidenceSnapshot.from_payload(
        source_type="news_article",
        ticker="aapl",
        natural_key="https://example.com/aapl-news",
        payload=b'{"headline":"services revenue rose"}',
        fetched_at=fetched_at,
        fetcher_name="web_search_tool",
        fetcher_version="1",
        storage_uri="artifacts/dataops/evidence/aapl-news.json",
        metadata={"publisher": "Example News"},
    )
    repeated = EvidenceSnapshot.from_payload(
        source_type="news_article",
        ticker="AAPL",
        natural_key="https://example.com/aapl-news",
        payload=b'{"headline":"services revenue rose"}',
        fetched_at=fetched_at,
        fetcher_name="web_search_tool",
        fetcher_version="1",
        storage_uri="artifacts/dataops/evidence/aapl-news.json",
        metadata={"publisher": "Example News"},
    )
    changed_payload = EvidenceSnapshot.from_payload(
        source_type="news_article",
        ticker="AAPL",
        natural_key="https://example.com/aapl-news",
        payload=b'{"headline":"services revenue fell"}',
        fetched_at=fetched_at,
        fetcher_name="web_search_tool",
        fetcher_version="1",
        storage_uri="artifacts/dataops/evidence/aapl-news-v2.json",
        metadata={"publisher": "Example News"},
    )

    assert repeated.snapshot_id == first.snapshot_id
    assert repeated.payload_hash == first.payload_hash
    assert changed_payload.payload_hash != first.payload_hash
    assert changed_payload.snapshot_id != first.snapshot_id
    assert first.ticker == "AAPL"


def test_evidence_snapshot_model_is_frozen() -> None:
    snapshot = EvidenceSnapshot.from_payload(
        source_type="market_quote",
        ticker="MSFT",
        natural_key="MSFT:2026-07-04T16:00:00Z",
        payload=b'{"price": 501.25}',
        fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        fetcher_name="stock_data_tool",
        fetcher_version="1",
        storage_uri="artifacts/dataops/evidence/msft-quote.json",
    )

    with pytest.raises(ValidationError):
        snapshot.ticker = "NVDA"  # type: ignore[misc]


def test_dataset_release_manifest_defaults_code_version_from_git_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dataops.contracts.current_code_version", lambda: "abc1234")

    manifest = DatasetReleaseManifest.create(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        release_type="evidence_snapshot",
        release_status="candidate",
        snapshot_ids=["news_article:abc"],
        config_hash="config-sha",
        quality_report_uri="artifacts/dataops/releases/evidence_snapshot/alpha-evidence/0.1.0/quality_report.json",
        artifact_uri="artifacts/dataops/releases/evidence_snapshot/alpha-evidence/0.1.0",
        parent_release_ids=[],
        created_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )

    assert manifest.code_version == "abc1234"
    assert manifest.release_id == "alpha-evidence:0.1.0"


def test_dataset_release_manifest_rejects_unknown_release_type() -> None:
    with pytest.raises(ValidationError):
        DatasetReleaseManifest.create(
            dataset_name="alpha-evidence",
            dataset_version="0.1.0",
            release_type="raw_dump",  # type: ignore[arg-type]
            release_status="candidate",
            snapshot_ids=[],
            config_hash="config-sha",
            code_version="abc1234",
            quality_report_uri="quality.json",
            artifact_uri="artifact",
            parent_release_ids=[],
            created_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        )
