from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dataops.contracts import SourceType
from dataops.evidence_release import create_evidence_release
from dataops.replay import load_evidence_release_records
from dataops.snapshot_writer import EvidenceSnapshotWriter


def _write_snapshot(
    writer: EvidenceSnapshotWriter,
    *,
    source_type: SourceType,
    ticker: str,
    natural_key: str,
    payload: dict,
) -> None:
    writer.write_json(
        source_type=source_type,
        ticker=ticker,
        natural_key=natural_key,
        payload=payload,
        fetcher_name=f"{source_type}_fetcher",
        fetcher_version="1",
        fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )


def test_load_evidence_release_records_returns_manifest_and_pinned_snapshots(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="news_article",
        ticker="AAPL",
        natural_key="https://news.example.com/aapl",
        payload={"title": "Apple news"},
    )
    _write_snapshot(
        writer,
        source_type="market_quote",
        ticker="AAPL",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={"ticker": "AAPL", "current_price": 220.5},
    )
    create_evidence_release(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tickers=["AAPL"],
        required_source_types=["market_quote", "news_article"],
    )

    bundle = load_evidence_release_records("alpha-evidence:0.1.0", registry_root=registry_root)

    assert bundle.manifest.release_id == "alpha-evidence:0.1.0"
    assert sorted(record.snapshot.source_type for record in bundle.records) == [
        "market_quote",
        "news_article",
    ]
    assert bundle.records_by_ticker["AAPL"][0].snapshot.ticker == "AAPL"


def test_load_evidence_release_records_rejects_missing_pinned_snapshot(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="market_quote",
        ticker="AAPL",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={"ticker": "AAPL", "current_price": 220.5},
    )
    manifest = create_evidence_release(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tickers=["AAPL"],
        required_source_types=["market_quote"],
    )
    for path in snapshot_root.rglob("*.json"):
        path.unlink()

    with pytest.raises(ValueError, match=f"missing snapshot_id: {manifest.snapshot_ids[0]}"):
        load_evidence_release_records("alpha-evidence:0.1.0", registry_root=registry_root)


def test_load_evidence_release_records_rejects_payload_hash_drift(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="market_quote",
        ticker="AAPL",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={"ticker": "AAPL", "current_price": 220.5},
    )
    create_evidence_release(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tickers=["AAPL"],
        required_source_types=["market_quote"],
    )
    snapshot_path = next(snapshot_root.rglob("*.json"))
    record = json.loads(snapshot_path.read_text())
    record["payload"]["current_price"] = 999.0
    snapshot_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="payload_hash mismatch"):
        load_evidence_release_records("alpha-evidence:0.1.0", registry_root=registry_root)
