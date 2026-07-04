from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dataops.contracts import DatasetReleaseManifest
from dataops.registry import ReleaseRegistry


def _manifest(version: str = "0.1.0") -> DatasetReleaseManifest:
    return DatasetReleaseManifest.create(
        dataset_name="alpha-evidence",
        dataset_version=version,
        release_type="evidence_snapshot",
        release_status="approved",
        snapshot_ids=["news_article:abc"],
        config_hash="config-sha",
        code_version="abc1234",
        quality_report_uri=f"artifacts/dataops/releases/evidence_snapshot/alpha-evidence/{version}/quality_report.json",
        artifact_uri=f"artifacts/dataops/releases/evidence_snapshot/alpha-evidence/{version}",
        parent_release_ids=[],
        created_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )


def test_registry_appends_and_reads_releases(tmp_path) -> None:
    registry = ReleaseRegistry(tmp_path)
    first = _manifest("0.1.0")
    second = _manifest("0.1.1")

    registry.append_release(first)
    registry.append_release(second)

    assert registry.releases_path.read_text(encoding="utf-8").count("\n") == 2
    assert registry.read_releases() == [first, second]
    assert registry.get_release("alpha-evidence", "0.1.1") == second


def test_registry_rejects_duplicate_release_id_without_rewriting(tmp_path) -> None:
    registry = ReleaseRegistry(tmp_path)
    manifest = _manifest()
    registry.append_release(manifest)

    with pytest.raises(ValueError, match="already exists"):
        registry.append_release(manifest)

    assert registry.read_releases() == [manifest]


def test_registry_writes_and_reads_active_yaml_pointer(tmp_path) -> None:
    registry = ReleaseRegistry(tmp_path)
    manifest = _manifest()
    registry.append_release(manifest)

    registry.pin_active("evidence_snapshot", manifest)

    pointer_path = tmp_path / "active" / "evidence_snapshot.yaml"
    assert pointer_path.read_text(encoding="utf-8") == (
        "dataset_name: alpha-evidence\n"
        "dataset_version: 0.1.0\n"
        "release_type: evidence_snapshot\n"
        "release_status: approved\n"
        "release_id: alpha-evidence:0.1.0\n"
    )
    assert registry.read_active("evidence_snapshot") == manifest
