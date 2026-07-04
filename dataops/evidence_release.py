from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dataops.contracts import DatasetReleaseManifest, EvidenceSnapshot, SourceType, payload_hash_for
from dataops.gates import gate_evidence_snapshot
from dataops.registry import ReleaseRegistry


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    snapshot: EvidenceSnapshot
    payload: dict[str, Any]
    path: Path

    @property
    def payload_hash(self) -> str:
        payload_text = json.dumps(self.payload, sort_keys=True, separators=(",", ":"), default=str)
        return payload_hash_for(payload_text)


def _release_file_stem(dataset_name: str, dataset_version: str) -> str:
    return f"{dataset_name}__{dataset_version}".replace("/", "_").replace(":", "_")


def _config_hash(
    *,
    tickers: Sequence[str],
    required_source_types: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "tickers": [ticker.upper() for ticker in tickers],
            "required_source_types": list(required_source_types),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload_hash_for(payload)


def load_snapshot_records(snapshot_root: Path | str) -> list[SnapshotRecord]:
    root = Path(snapshot_root)
    if not root.exists():
        return []

    records: list[SnapshotRecord] = []
    for path in sorted(root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            SnapshotRecord(
                snapshot=EvidenceSnapshot.model_validate(raw["snapshot"]),
                payload=raw["payload"],
                path=path,
            )
        )
    return records


def _source_counts(
    records: list[SnapshotRecord],
    *,
    tickers: Sequence[str],
    source_types: Sequence[str],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        ticker: {source_type: 0 for source_type in source_types} for ticker in tickers
    }
    for record in records:
        ticker_counts = counts.get(record.snapshot.ticker)
        if ticker_counts is None:
            continue
        ticker_counts[record.snapshot.source_type] = ticker_counts.get(record.snapshot.source_type, 0) + 1
    return counts


def _release_errors(
    *,
    records: list[SnapshotRecord],
    tickers: Sequence[str],
    required_source_types: Sequence[str],
    manifest: DatasetReleaseManifest,
) -> list[str]:
    errors: list[str] = []
    if not records:
        errors.append("snapshot release must include at least one snapshot")

    for record in records:
        if record.payload_hash != record.snapshot.payload_hash:
            errors.append(
                f"payload_hash mismatch for {record.path}: "
                f"expected {record.snapshot.payload_hash}, got {record.payload_hash}"
            )

    counts = _source_counts(records, tickers=tickers, source_types=required_source_types)
    for ticker in tickers:
        missing = [
            source_type
            for source_type in required_source_types
            if counts.get(ticker, {}).get(source_type, 0) == 0
        ]
        if missing:
            errors.append(f"missing snapshots for {ticker}: {', '.join(missing)}")

    gate = gate_evidence_snapshot(manifest, [record.snapshot for record in records])
    errors.extend(gate.errors)
    return errors


def _write_quality_report(
    *,
    registry_root: Path,
    dataset_name: str,
    dataset_version: str,
    passed: bool,
    errors: list[str],
    records: list[SnapshotRecord],
    tickers: Sequence[str],
    required_source_types: Sequence[str],
) -> Path:
    report_path = (
        registry_root
        / "quality_reports"
        / f"{_release_file_stem(dataset_name, dataset_version)}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "errors": errors,
                "ticker_universe": list(tickers),
                "required_source_types": list(required_source_types),
                "snapshot_count": len(records),
                "source_counts": _source_counts(
                    records,
                    tickers=tickers,
                    source_types=required_source_types,
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path


def create_evidence_release(
    *,
    dataset_name: str,
    dataset_version: str,
    snapshot_root: Path | str,
    registry_root: Path | str = Path("artifacts/dataops"),
    tickers: Sequence[str],
    required_source_types: Sequence[SourceType],
    pointer_name: str = "evidence",
) -> DatasetReleaseManifest:
    ticker_universe = sorted({ticker.upper() for ticker in tickers})
    source_types = list(required_source_types)
    registry_path = Path(registry_root)
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record.snapshot.ticker in ticker_universe
    ]
    snapshot_ids = sorted(record.snapshot.snapshot_id for record in records)
    report_path = (
        registry_path
        / "quality_reports"
        / f"{_release_file_stem(dataset_name, dataset_version)}.json"
    )
    manifest = DatasetReleaseManifest.create(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        release_type="evidence_snapshot",
        release_status="approved",
        snapshot_ids=snapshot_ids,
        config_hash=_config_hash(
            tickers=ticker_universe,
            required_source_types=source_types,
        ),
        quality_report_uri=report_path.as_posix(),
        artifact_uri=Path(snapshot_root).as_posix(),
        parent_release_ids=[],
    )

    errors = _release_errors(
        records=records,
        tickers=ticker_universe,
        required_source_types=source_types,
        manifest=manifest,
    )
    _write_quality_report(
        registry_root=registry_path,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        passed=not errors,
        errors=errors,
        records=records,
        tickers=ticker_universe,
        required_source_types=source_types,
    )
    if errors:
        raise ValueError("; ".join(errors))

    registry = ReleaseRegistry(registry_path)
    registry.append_release(manifest)
    registry.pin_active(pointer_name, manifest)
    return manifest
