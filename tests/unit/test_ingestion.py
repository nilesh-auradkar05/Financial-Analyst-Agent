from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import app.components.retrieval.ingestion as ingestion


class StubVectorStore:
    def __init__(self) -> None:
        self.documents: list[Any] = []
        self.deleted_tickers: list[str] = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        return len(documents)

    def delete_by_ticker(self, ticker: str) -> int:
        self.deleted_tickers.append(ticker.upper())
        kept_documents = [
            doc for doc in self.documents if doc.metadata.get("ticker") != ticker.upper()
        ]
        deleted_count = len(self.documents) - len(kept_documents)
        self.documents = kept_documents
        return deleted_count

    def count_documents(self, filters=None):
        ticker = getattr(filters, "ticker", None)
        if ticker is None:
            return len(self.documents)
        return sum(
            1 for doc in self.documents if doc.metadata.get("ticker") == ticker.upper()
        )

    def get_stats(self):
        return {"document_count": len(self.documents)}


@pytest.fixture
def sample_filing():
    return SimpleNamespace(
        metadata=SimpleNamespace(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2025-09-28",
            company_name="Apple Inc.",
            accession_number="0000320193-25-000001",
            source_url="https://www.sec.gov/Archives/example",
        ),
        sections={
            "Business": SimpleNamespace(
                content=("Apple designs hardware, software, and services. " * 80)
            ),
            "Item 1A. Risk Factors": SimpleNamespace(
                content=(
                    "The company is exposed to supply-chain and regulatory risk. "
                    * 90
                )
            ),
            "Management's Discussion and Analysis": SimpleNamespace(
                content=(
                    "Revenue, margin, and services mix changed during the fiscal year. "
                    * 90
                )
            ),
        },
    )


def test_canonicalize_section_name_maps_common_sec_variants():
    business = ingestion.canonicalize_section_name("Business")
    risk_factors = ingestion.canonicalize_section_name("Item 1A. Risk Factors")
    mda = ingestion.canonicalize_section_name("Management's Discussion and Analysis")
    market_risk = ingestion.canonicalize_section_name(
        "Item 7A Quantitative and Qualitative Disclosures About Market Risk"
    )

    assert business is not None
    assert risk_factors is not None
    assert mda is not None
    assert market_risk is not None
    assert business.key == "business"
    assert risk_factors.key == "risk_factors"
    assert mda.key == "md&a"
    assert market_risk.key == "market_risk"


def test_build_index_documents_emits_metadata_rich_documents(sample_filing):
    result = ingestion.build_index_documents(sample_filing)

    assert result.sections_requested == ["business", "risk_factors", "md&a", "market_risk"]
    assert result.sections_found == ["business", "risk_factors", "md&a"]
    assert result.sections_skipped == ["market_risk"]
    assert result.documents

    first_document = result.documents[0]
    metadata = first_document.metadata
    assert metadata["ticker"] == "AAPL"
    assert metadata["company_name"] == "Apple Inc."
    assert metadata["filing_type"] == "10-K"
    assert metadata["filing_date"] == "2025-09-28"
    assert metadata["accession_number"] == "0000320193-25-000001"
    assert metadata["source_url"] == "https://www.sec.gov/Archives/example"
    assert metadata["section_name"] == "Business"
    assert metadata["section_key"] == "business"
    assert metadata["document_id"] == "AAPL_10-K_2025-09-28"
    assert metadata["parent_section_id"] == "AAPL_10-K_2025-09-28:business"
    assert metadata["chunk_id"] == first_document.id
    assert metadata["chunk_index"] == 0
    assert first_document.id.startswith("AAPL_10-K_2025-09-28_business_")


def test_ingest_filing_writes_section_aware_documents(monkeypatch, sample_filing):
    store = StubVectorStore()
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    result = ingestion.ingest_filing(sample_filing, replace_existing=True)

    assert result.success is True
    assert result.sections_processed == ["business", "risk_factors", "md&a"]
    assert result.sections_requested == ["business", "risk_factors", "md&a", "market_risk"]
    assert result.sections_found == ["business", "risk_factors", "md&a"]
    assert result.sections_skipped == ["market_risk"]
    assert result.documents_written == result.total_chunks
    assert result.total_chunks == len(store.documents)
    assert store.deleted_tickers == ["AAPL"]


def test_ingest_filing_short_circuits_when_existing_documents_present(
    monkeypatch,
    sample_filing,
):
    store = StubVectorStore()
    store.documents.append(
        SimpleNamespace(
            id="AAPL_10-K_2025-09-28_business_000",
            text="existing business chunk",
            metadata={"ticker": "AAPL"},
        )
    )
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    result = ingestion.ingest_filing(sample_filing, replace_existing=False)

    assert result.success is True
    assert result.total_chunks == 1
    assert result.documents_written == 0
    assert result.sections_processed == []
    assert len(store.documents) == 1
