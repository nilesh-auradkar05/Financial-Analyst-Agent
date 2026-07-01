from __future__ import annotations

import app.components.retrieval.ingestion as ingestion
from app.components.retrieval.vector_store import SearchFilters

CRITICAL_SECTIONS = ["business", "risk_factors", "md&a", "market_risk"]


def test_representative_filings_have_critical_section_coverage_through_store(
    monkeypatch,
    representative_filings,
    store,
) -> None:
    """test-plan §3: present critical sections yield chunks through store interface."""

    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    for filing in representative_filings:
        result = ingestion.ingest_filing(filing)

        assert result.success is True
        assert set(result.sections_found) == set(CRITICAL_SECTIONS)
        assert result.sections_skipped == []

    for filing in representative_filings:
        ticker = filing.metadata.ticker.upper()
        accession_number = filing.metadata.accession_number
        for section_key in CRITICAL_SECTIONS:
            count = store.count_documents(
                SearchFilters(
                    ticker=ticker,
                    filing_type="10-K",
                    section_key=section_key,
                    extra={"accession_number": accession_number},
                )
            )

            assert count > 0, f"{ticker} missing {section_key} chunks"
