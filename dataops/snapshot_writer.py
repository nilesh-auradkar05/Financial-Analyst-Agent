from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataops.contracts import EvidenceSnapshot, SourceType, payload_hash_for, snapshot_id_for


class EvidenceSnapshotWriter:
    def __init__(self, root: Path | str = Path("artifacts/dataops/evidence_snapshots")) -> None:
        self.root = Path(root)

    def write_json(
        self,
        *,
        source_type: SourceType,
        ticker: str,
        natural_key: str,
        payload: dict[str, Any],
        fetcher_name: str,
        fetcher_version: str,
        fetched_at: datetime | None = None,
        metadata: dict[str, str] | None = None,
    ) -> EvidenceSnapshot:
        fetched_at = fetched_at or datetime.now(timezone.utc)
        payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        payload_hash = payload_hash_for(payload_text)
        snapshot_id = snapshot_id_for(source_type, natural_key, payload_hash)
        ticker_upper = ticker.upper()
        storage_path = self.root / source_type / ticker_upper / f"{snapshot_id}.json"

        snapshot = EvidenceSnapshot(
            snapshot_id=snapshot_id,
            source_type=source_type,
            ticker=ticker_upper,
            natural_key=natural_key,
            payload_hash=payload_hash,
            fetched_at=fetched_at,
            fetcher_name=fetcher_name,
            fetcher_version=fetcher_version,
            storage_uri=storage_path.as_posix(),
            metadata=metadata or {},
        )

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_text(
            json.dumps(
                {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return snapshot
