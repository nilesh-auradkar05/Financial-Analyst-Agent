from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dataops.contracts import SourceType
from dataops.evidence_release import create_evidence_release
from dataops.registry import ReleaseRegistry
from dataops.snapshot_writer import EvidenceSnapshotWriter


def _write_snapshot(
    writer: EvidenceSnapshotWriter,
    *,
    source_type: SourceType,
    ticker: str = "AAPL",
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


def test_create_evidence_release_registers_and_pins_approved_manifest(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="news_article",
        natural_key="https://news.example.com/aapl",
        payload={"title": "Apple news", "snippet": "Services revenue rose."},
    )
    _write_snapshot(
        writer,
        source_type="market_quote",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={"ticker": "AAPL", "current_price": 220.5},
    )

    manifest = create_evidence_release(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tickers=["aapl"],
        required_source_types=["market_quote", "news_article"],
        pointer_name="evidence",
    )

    registry = ReleaseRegistry(registry_root)
    active = registry.read_active("evidence")
    report = json.loads((registry_root / "quality_reports" / "alpha-evidence__0.1.0.json").read_text())

    assert manifest.release_id == "alpha-evidence:0.1.0"
    assert manifest.release_type == "evidence_snapshot"
    assert manifest.release_status == "approved"
    assert active == manifest
    assert manifest.snapshot_ids == sorted(manifest.snapshot_ids)
    assert manifest.artifact_uri == snapshot_root.as_posix()
    assert manifest.quality_report_uri.endswith("quality_reports/alpha-evidence__0.1.0.json")
    assert report["passed"] is True
    assert report["ticker_universe"] == ["AAPL"]
    assert report["source_counts"]["AAPL"]["market_quote"] == 1
    assert report["source_counts"]["AAPL"]["news_article"] == 1


def test_create_evidence_release_rejects_snapshot_payload_drift(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="market_quote",
        natural_key="AAPL:2026-07-04T00:00:00+00:00",
        payload={"ticker": "AAPL", "current_price": 220.5},
    )
    snapshot_path = next(snapshot_root.rglob("*.json"))
    record = json.loads(snapshot_path.read_text())
    record["payload"]["current_price"] = 999.0
    snapshot_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="payload_hash mismatch"):
        create_evidence_release(
            dataset_name="alpha-evidence",
            dataset_version="0.1.0",
            snapshot_root=snapshot_root,
            registry_root=registry_root,
            tickers=["AAPL"],
            required_source_types=["market_quote"],
        )

    assert not (registry_root / "releases.jsonl").exists()


def test_create_evidence_release_requires_ticker_source_coverage(tmp_path) -> None:
    snapshot_root = tmp_path / "snapshots"
    registry_root = tmp_path / "registry"
    writer = EvidenceSnapshotWriter(snapshot_root)
    _write_snapshot(
        writer,
        source_type="news_article",
        natural_key="https://news.example.com/aapl",
        payload={"title": "Apple news"},
    )

    with pytest.raises(ValueError, match="missing snapshots for AAPL: market_quote"):
        create_evidence_release(
            dataset_name="alpha-evidence",
            dataset_version="0.1.0",
            snapshot_root=snapshot_root,
            registry_root=registry_root,
            tickers=["AAPL"],
            required_source_types=["market_quote", "news_article"],
        )

    assert not (registry_root / "releases.jsonl").exists()
