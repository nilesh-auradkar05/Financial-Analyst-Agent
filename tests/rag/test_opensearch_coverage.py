"""Ingestion coverage and idempotency against the OpenSearch store (test-plan §5; S3-T05)."""

from __future__ import annotations

from typing import Any

import app.components.retrieval.ingestion as ingestion
from app.components.retrieval.vector_store import SearchFilters
from tests.filings import CRITICAL_SECTIONS, make_filing
from tests.rag.helpers import all_chunk_ids, raw_document


def test_representative_filings_have_critical_section_coverage_under_opensearch(
    monkeypatch, make_store
) -> None:
    """test-plan §5: Coverage under OpenSearch -> S1 section coverage passes against the
    container."""
    store = make_store()
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)
    representative_filings: list[Any] = [
        make_filing("AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28"),
        make_filing("MSFT", accession_number="0000789019-25-000001", filing_date="2025-06-30"),
        make_filing("NVDA", accession_number="0001045810-25-000001", filing_date="2025-01-26"),
    ]

    for filing in representative_filings:
        result = ingestion.ingest_filing(filing)

        assert result.success is True
        assert set(result.sections_found) == set(CRITICAL_SECTIONS)
        assert result.sections_skipped == []

    for filing in representative_filings:
        ticker = filing.metadata.ticker.upper()
        for section_key in CRITICAL_SECTIONS:
            count = store.count_documents(
                SearchFilters(
                    ticker=ticker,
                    filing_type="10-K",
                    section_key=section_key,
                    extra={"accession_number": filing.metadata.accession_number},
                )
            )

            assert count > 0, f"{ticker} missing {section_key} chunks"


def test_reingesting_same_filing_is_idempotent_under_opensearch(
    monkeypatch, opensearch_url, index_name, make_store
) -> None:
    """test-plan §5: Coverage under OpenSearch -> idempotency passes against the container
    (same filing yields stable chunk ids and no duplicates)."""
    store = make_store(index_name)
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)
    sample_filing: Any = make_filing(
        "AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28"
    )

    first = ingestion.ingest_filing(sample_filing)
    first_ids = sorted(all_chunk_ids(store))
    second = ingestion.ingest_filing(sample_filing)
    second_ids = sorted(all_chunk_ids(store))

    assert first.success is True
    assert second.success is True
    assert first_ids
    assert second_ids == first_ids
    assert len(second_ids) == len(set(second_ids))
    assert store.count_documents() == len(first_ids)
    for chunk_id in first_ids:
        status, stored = raw_document(opensearch_url, index_name, chunk_id)
        assert status == 200 and stored["_id"] == chunk_id


def test_replace_existing_rewrites_one_ticker_without_duplicates_or_collateral(
    monkeypatch, make_store
) -> None:
    """test-plan §5: Coverage under OpenSearch -> replacement path stays stable and
    duplicate-free, and leaves other tickers untouched."""
    store = make_store()
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)
    apple: Any = make_filing("AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28")
    microsoft: Any = make_filing(
        "MSFT", accession_number="0000789019-25-000001", filing_date="2025-06-30"
    )
    ingestion.ingest_filing(apple)
    ingestion.ingest_filing(microsoft)
    ids_before = sorted(all_chunk_ids(store))
    microsoft_before = store.count_documents(SearchFilters(ticker="MSFT"))

    result = ingestion.ingest_filing(apple, replace_existing=True)

    ids_after = sorted(all_chunk_ids(store))
    assert result.success is True
    assert ids_after == ids_before
    assert len(ids_after) == len(set(ids_after))
    assert store.count_documents() == len(ids_before)
    assert store.count_documents(SearchFilters(ticker="MSFT")) == microsoft_before
