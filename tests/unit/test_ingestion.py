from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.components.retrieval.ingestion as ingestion
import app.services.tools.sec_filings_tool as sec_tool


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
    assert metadata["document_id"] == "0000320193-25-000001"
    assert metadata["parent_section_id"] == "0000320193-25-000001:business"
    assert metadata["chunk_id"] == first_document.id
    assert metadata["chunk_index"] == 0
    assert first_document.id.startswith("0000320193-25-000001_business_")


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


# Trace: docs/test-plan.md §16, Ingest no 10-K.
@pytest.mark.asyncio
async def test_ingest_10k_distinguishes_missing_filing_from_upstream_failure(monkeypatch):
    async def absent(_ticker):
        return None

    monkeypatch.setattr(ingestion, "get_latest_10k", absent)
    missing = await ingestion.ingest_10k_for_ticker(" aapl ")
    assert missing.success is False
    assert missing.error_code == "filing_not_found"
    assert missing.error == "No 10-K filings found for AAPL."

    async def unavailable(_ticker):
        raise RuntimeError("private upstream details")

    monkeypatch.setattr(ingestion, "get_latest_10k", unavailable)
    failure = await ingestion.ingest_10k_for_ticker("AAPL")
    assert failure.success is False
    assert failure.error_code != "filing_not_found"
    assert "private" not in str(failure.error)


# Trace: docs/test-plan.md §16, Ingest no 10-K.
@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_fails", [False, True])
async def test_sec_lookup_distinguishes_no_filing_from_upstream_failure(monkeypatch, lookup_fails):
    async def no_edgartools(_ticker):
        return None

    def respond(request):
        if request.url.path.endswith("/CIKAAPL.json"):
            return httpx.Response(200, json={"cik": 123})
        recent = {"form": ["10-K"]} if lookup_fails else {"form": []}
        return httpx.Response(200, json={"filings": {"recent": recent}})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(sec_tool, "_get_latest_10k_with_edgartools", no_edgartools)
    monkeypatch.setattr(
        sec_tool.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )

    result = await ingestion.ingest_10k_for_ticker("AAPL")
    assert result.success is False
    assert result.error_code == (None if lookup_fails else "filing_not_found")
    assert result.error == ("10-K ingestion failed." if lookup_fails else "No 10-K filings found for AAPL.")
