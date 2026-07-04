from __future__ import annotations

from dataclasses import dataclass

from dataops.contracts import DatasetReleaseManifest, EvidenceSnapshot, snapshot_id_for


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    errors: list[str]


def gate_evidence_snapshot(
    manifest: DatasetReleaseManifest,
    snapshots: list[EvidenceSnapshot],
) -> QualityGateResult:
    errors: list[str] = []

    if manifest.release_type != "evidence_snapshot":
        errors.append("manifest release_type must be evidence_snapshot")

    if not manifest.snapshot_ids:
        errors.append("snapshot_ids must not be empty")

    snapshots_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    for snapshot_id in manifest.snapshot_ids:
        if snapshot_id not in snapshots_by_id:
            errors.append(f"manifest references missing snapshot_id: {snapshot_id}")

    natural_keys: dict[tuple[str, str], str] = {}
    for snapshot in snapshots:
        if not snapshot.storage_uri:
            errors.append(f"storage_uri must not be empty for {snapshot.snapshot_id}")
        if not snapshot.natural_key:
            errors.append(f"natural_key must not be empty for {snapshot.snapshot_id}")

        expected_id = snapshot_id_for(
            snapshot.source_type,
            snapshot.natural_key,
            snapshot.payload_hash,
        )
        if snapshot.snapshot_id != expected_id:
            errors.append(f"snapshot_id does not match payload hash for {snapshot.snapshot_id}")

        key = (snapshot.source_type, snapshot.natural_key)
        previous_hash = natural_keys.get(key)
        if previous_hash is not None and previous_hash != snapshot.payload_hash:
            errors.append(
                "duplicate natural key with different payload hash: "
                f"{snapshot.source_type} {snapshot.natural_key}"
            )
        natural_keys[key] = snapshot.payload_hash

    return QualityGateResult(passed=not errors, errors=errors)
