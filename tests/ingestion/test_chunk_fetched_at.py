"""Trace: docs/test-plan.md section 13 row "Ingestion records fetch time (2026-10-11)";
docs/SPEC.md 7.y (live filings count from the fetched_at recorded at ingestion).

Every chunk written by ingestion carries an ISO-8601 `fetched_at` with a UTC offset; chunk ids
and the other metadata are unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import app.components.retrieval.ingestion as ingestion

# Metadata keys the canonical-metadata test (test_chunk_metadata.py) asserts on.
DOCUMENTED_KEYS = {
    "chunk_id",
    "chunk_index",
    "section_key",
    "ticker",
    "filing_type",
    "filing_date",
    "accession_number",
    "source_url",
}


def test_every_chunk_records_a_fetched_at_with_a_utc_offset(sample_filing) -> None:
    before = datetime.now(timezone.utc)
    result = ingestion.build_index_documents(sample_filing)
    after = datetime.now(timezone.utc)

    assert result.documents
    for document in result.documents:
        fetched_at = datetime.fromisoformat(document.metadata["fetched_at"])
        assert fetched_at.utcoffset() == timedelta(0)
        assert before <= fetched_at <= after


def test_explicit_fetched_at_is_stored_as_that_instant(sample_filing) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    instant = datetime(2026, 6, 20, 17, 30, tzinfo=ist)

    result = ingestion.build_index_documents(sample_filing, fetched_at=instant)

    assert result.documents
    for document in result.documents:
        stored = datetime.fromisoformat(document.metadata["fetched_at"])
        assert stored.utcoffset() is not None
        assert stored == instant


def test_fetched_at_changes_nothing_but_itself(sample_filing) -> None:
    earlier = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
    later = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)

    first = ingestion.build_index_documents(sample_filing, fetched_at=earlier)
    second = ingestion.build_index_documents(sample_filing, fetched_at=later)

    assert [d.id for d in first.documents] == [d.id for d in second.documents]
    assert [d.text for d in first.documents] == [d.text for d in second.documents]
    for a, b in zip(first.documents, second.documents, strict=True):
        assert a.metadata["fetched_at"] != b.metadata["fetched_at"]
        assert {k: v for k, v in a.metadata.items() if k != "fetched_at"} == {
            k: v for k, v in b.metadata.items() if k != "fetched_at"
        }
        assert DOCUMENTED_KEYS <= set(a.metadata)
        assert a.metadata["chunk_id"] == a.id
