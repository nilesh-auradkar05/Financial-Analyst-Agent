from __future__ import annotations

from types import SimpleNamespace

import pytest

import rag.ingestion as ingestion


class StubCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def get(self, where=None, include=None):
        ticker = (where or {}).get("ticker")
        ids = [doc_id for doc_id in self.ids if ticker is None or doc_id.startswith(ticker)]
        return {"ids": ids}

class StubVectorStore:
    def __init__(self) -> None:
        self._collection = StubCollection()
        self.documents = []
        self.deleted_tickers: list[str] = []

    def add_documents(self, documents=None, *, texts=None, metadatas=None, ids=None):
        if documents is not None:
            self.documents.extend(documents)
            self._collection.ids.extend([document.id for document in documents])
            return len(documents)

        payload = []
        for text, metadata, doc_id in zip(texts or [], metadatas or [], ids or []):
            payload.append(SimpleNamespace(id=doc_id, text=text, metadata=metadata))
        self.documents.extend(payload)
        self._collection.ids.extend([document.id for document in payload])
        return len(payload)

    def delete_by_ticker(self, ticker: str) -> int:
        self.deleted_tickers.append(ticker.upper())
        kept_documents = [doc for doc in self.documents if doc.metadata.get("ticker") != ticker.upper()]
        deleted_count = len(self.documents) - len(kept_documents)
        self.documents = kept_documents
        self._collection.ids = [doc.id for doc in self.documents]
        return deleted_count

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
            "Business": SimpleNamespace(content=("Apple designs hardware, software, and services. " * 80)),
            "Item 1A. Risk Factors": SimpleNamespace(content=("The company is exposed to supply-chain and regulatory risk. " * 90)),
            "Management's Discussion and Analysis": SimpleNamespace(content=("Revenue, margin, and services mix changed during the fiscal year. " * 90)),
        },
    )

def test_canonicalize_section_name_maps_common_sec_variants():
    assert ingestion.canonicalize_section_name("Business").key == "business"
    assert ingestion.canonicalize_section_name("Item 1A. Risk Factors").key == "risk_factors"
    assert ingestion.canonicalize_section_name("Management's Discussion and Analysis").key == "md&a"
    assert ingestion.canonicalize_section_name("Item 7A Quantitative and Qualitative Disclosures About Market Risk").key == "market_risk"

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

def test_ingest_filing_short_circuits_when_existing_documents_present(monkeypatch, sample_filing):
    store = StubVectorStore()
    store._collection.ids = ["AAPL_10-K_2025-09-28_business_000"]
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    result = ingestion.ingest_filing(sample_filing, replace_existing=False)

    assert result.success is True
    assert result.total_chunks == 1
    assert result.documents_written == 0
    assert result.sections_processed == []
    assert store.documents == []
