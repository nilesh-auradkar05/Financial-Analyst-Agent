from __future__ import annotations

from datetime import datetime, timezone

from dataops.contracts import DatasetReleaseManifest, EvidenceSnapshot
from dataops.gates import gate_evidence_snapshot


def _snapshot(
    *,
    natural_key: str = "https://example.com/aapl-news",
    payload: bytes = b'{"headline":"services revenue rose"}',
) -> EvidenceSnapshot:
    return EvidenceSnapshot.from_payload(
        source_type="news_article",
        ticker="AAPL",
        natural_key=natural_key,
        payload=payload,
        fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        fetcher_name="web_search_tool",
        fetcher_version="1",
        storage_uri="artifacts/dataops/evidence/aapl-news.json",
    )


def _manifest(snapshot_ids: list[str]) -> DatasetReleaseManifest:
    return DatasetReleaseManifest.create(
        dataset_name="alpha-evidence",
        dataset_version="0.1.0",
        release_type="evidence_snapshot",
        release_status="candidate",
        snapshot_ids=snapshot_ids,
        config_hash="config-sha",
        code_version="abc1234",
        quality_report_uri="quality.json",
        artifact_uri="artifact",
        parent_release_ids=[],
        created_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )


def test_evidence_snapshot_gate_passes_valid_manifest_and_snapshots() -> None:
    snapshot = _snapshot()
    manifest = _manifest([snapshot.snapshot_id])

    result = gate_evidence_snapshot(manifest, [snapshot])

    assert result.passed is True
    assert result.errors == []


def test_evidence_snapshot_gate_requires_snapshot_ids() -> None:
    result = gate_evidence_snapshot(_manifest([]), [])

    assert result.passed is False
    assert "snapshot_ids must not be empty" in result.errors


def test_evidence_snapshot_gate_rejects_mutated_payload_under_existing_id() -> None:
    original = _snapshot()
    mutated = original.model_copy(update={"payload_hash": "0" * 64})
    manifest = _manifest([original.snapshot_id])

    result = gate_evidence_snapshot(manifest, [mutated])

    assert result.passed is False
    assert f"snapshot_id does not match payload hash for {original.snapshot_id}" in result.errors


def test_evidence_snapshot_gate_rejects_duplicate_natural_key_with_different_hash() -> None:
    first = _snapshot()
    second = _snapshot(payload=b'{"headline":"services revenue fell"}')
    manifest = _manifest([first.snapshot_id, second.snapshot_id])

    result = gate_evidence_snapshot(manifest, [first, second])

    assert result.passed is False
    assert "duplicate natural key with different payload hash: news_article https://example.com/aapl-news" in result.errors
