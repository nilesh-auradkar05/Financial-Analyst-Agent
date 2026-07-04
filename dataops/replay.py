from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataops.contracts import DatasetReleaseManifest
from dataops.evidence_release import SnapshotRecord, load_snapshot_records
from dataops.registry import ReleaseRegistry


@dataclass(frozen=True, slots=True)
class EvidenceReplayBundle:
    manifest: DatasetReleaseManifest
    records: list[SnapshotRecord]

    @property
    def records_by_ticker(self) -> dict[str, list[SnapshotRecord]]:
        grouped: dict[str, list[SnapshotRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.snapshot.ticker, []).append(record)
        return grouped


def _parse_release_id(release_id: str) -> tuple[str, str]:
    if ":" not in release_id:
        raise ValueError("release id must be formatted as <name:version>")
    dataset_name, dataset_version = release_id.split(":", 1)
    if not dataset_name or not dataset_version:
        raise ValueError("release id must be formatted as <name:version>")
    return dataset_name, dataset_version


def load_evidence_release_records(
    release_id: str,
    *,
    registry_root: Path | str = Path("artifacts/dataops"),
) -> EvidenceReplayBundle:
    dataset_name, dataset_version = _parse_release_id(release_id)
    registry = ReleaseRegistry(registry_root)
    manifest = registry.get_release(dataset_name, dataset_version)
    if manifest is None:
        raise ValueError(f"release not found: {release_id}")
    if manifest.release_type != "evidence_snapshot":
        raise ValueError(f"release is not evidence_snapshot: {release_id}")

    records_by_id = {
        record.snapshot.snapshot_id: record
        for record in load_snapshot_records(manifest.artifact_uri)
    }
    selected: list[SnapshotRecord] = []
    errors: list[str] = []
    for snapshot_id in manifest.snapshot_ids:
        record = records_by_id.get(snapshot_id)
        if record is None:
            errors.append(f"missing snapshot_id: {snapshot_id}")
            continue
        if record.payload_hash != record.snapshot.payload_hash:
            errors.append(f"payload_hash mismatch for {record.path}")
            continue
        selected.append(record)

    if errors:
        raise ValueError("; ".join(errors))
    return EvidenceReplayBundle(manifest=manifest, records=selected)
