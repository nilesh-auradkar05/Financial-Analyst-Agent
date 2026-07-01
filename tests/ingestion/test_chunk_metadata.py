from __future__ import annotations

import app.components.retrieval.ingestion as ingestion
from app.services.tools.sec_filings_tool import Filing, FilingMetaData, FilingSection


def test_chunk_metadata_is_canonical_and_identity_is_accession_based(sample_filing) -> None:
    """test-plan §3: metadata is normalized and chunk_id is accession/section/index."""

    result = ingestion.build_index_documents(sample_filing)

    assert result.documents
    assert result.sections_requested == ["business", "risk_factors", "md&a", "market_risk"]
    assert set(result.sections_found) == {"business", "risk_factors", "md&a", "market_risk"}
    assert result.sections_skipped == []

    for document in result.documents:
        metadata = document.metadata
        chunk_index = metadata["chunk_index"]
        section_key = metadata["section_key"]
        expected_chunk_id = f"0000320193-25-000001_{section_key}_{chunk_index:03d}"

        assert document.id == expected_chunk_id
        assert metadata["chunk_id"] == expected_chunk_id
        assert metadata["ticker"] == "AAPL"
        assert metadata["filing_type"] == "10-K"
        assert metadata["filing_date"] == "2025-09-28"
        assert metadata["accession_number"] == "0000320193-25-000001"
        assert metadata["source_url"] == "https://www.sec.gov/Archives/0000320193-25-000001"
        assert metadata["section_key"] in {"business", "risk_factors", "md&a", "market_risk"}


def test_missing_section_is_reported_not_silently_ignored() -> None:
    """test-plan §3: absent sections are explicit in ingestion metadata."""

    filing = Filing(
        metadata=FilingMetaData(
            cik="0000320193",
            accession_number="0000320193-25-000001",
            filing_type="10-K",
            filing_date="2025-09-28",
            primary_document="aapl-20250928.htm",
            ticker="AAPL",
        ),
        sections={
            "Item 1. Business": FilingSection(
                name="Item 1. Business",
                content="Apple business disclosure. " * 80,
            ),
        },
    )

    result = ingestion.build_index_documents(filing)

    assert result.sections_found == ["business"]
    assert result.sections_skipped == ["risk_factors", "md&a", "market_risk"]
