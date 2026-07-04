from __future__ import annotations

import json
from datetime import datetime, timezone

from dataops.snapshot_writer import EvidenceSnapshotWriter


def test_snapshot_writer_persists_payload_with_snapshot_metadata(tmp_path) -> None:
    writer = EvidenceSnapshotWriter(tmp_path)

    snapshot = writer.write_json(
        source_type="news_article",
        ticker="AAPL",
        natural_key="https://example.com/aapl",
        payload={"title": "Apple news", "body": "Services revenue rose."},
        fetcher_name="web_search_tool",
        fetcher_version="1",
        fetched_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        metadata={"publisher": "Example"},
    )

    record = json.loads((tmp_path / "news_article" / "AAPL" / f"{snapshot.snapshot_id}.json").read_text())

    assert snapshot.storage_uri.endswith(f"{snapshot.snapshot_id}.json")
    assert record["snapshot"]["snapshot_id"] == snapshot.snapshot_id
    assert record["snapshot"]["payload_hash"] == snapshot.payload_hash
    assert record["payload"] == {"title": "Apple news", "body": "Services revenue rose."}
