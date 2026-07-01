from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

CRITICAL_SECTIONS = ["business", "risk_factors", "md&a", "market_risk"]


class InMemoryRetrievalStore:
    """Faithful store double: upserts by chunk id and filters by metadata."""

    def __init__(self) -> None:
        self.documents: dict[str, IndexDocument] = {}
        self.deleted_tickers: list[str] = []

    @property
    def count(self) -> int:
        return len(self.documents)

    def add_documents(self, documents: list[IndexDocument]) -> int:
        for document in documents:
            self.documents[document.id] = document
        return len(documents)

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        n_results: int = 5,
    ) -> SearchResult:
        del query
        matches = self._matching_documents(filters)
        chunks = [
            RetrievedChunk(
                id=document.id,
                text=document.text,
                metadata=document.metadata,
                distance=0.0,
            )
            for document in matches[:n_results]
        ]
        return SearchResult(
            query="in-memory",
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=0.0,
            filter_used=filters.to_backend_filter() if filters else None,
        )

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        n_results: int = 8,
        query: str | None = None,
        filing_type: str | None = None,
    ) -> SearchResult:
        del query
        chunks: list[RetrievedChunk] = []
        for section_key in sections:
            result = self.search(
                f"{ticker} {section_key}",
                filters=SearchFilters(
                    ticker=ticker,
                    filing_type=filing_type,
                    section_key=section_key,
                ),
                n_results=n_results,
            )
            chunks.extend(result.chunks)
        return SearchResult(
            query=f"{ticker} sections",
            chunks=chunks[:n_results],
            total_results=min(len(chunks), n_results),
            search_time_ms=0.0,
            filter_used={"ticker": ticker.upper(), "sections": sections},
        )

    def search_by_ticker(
        self,
        query: str,
        ticker: str,
        n_results: int = 5,
        section: str | None = None,
    ) -> SearchResult:
        return self.search(
            query,
            filters=SearchFilters(ticker=ticker, section_key=section),
            n_results=n_results,
        )

    def delete_by_ticker(self, ticker: str) -> int:
        normalized_ticker = ticker.upper()
        self.deleted_tickers.append(normalized_ticker)
        matching_ids = [
            document_id
            for document_id, document in self.documents.items()
            if document.metadata.get("ticker") == normalized_ticker
        ]
        for document_id in matching_ids:
            del self.documents[document_id]
        return len(matching_ids)

    def count_documents(self, filters: SearchFilters | None = None) -> int:
        return len(self._matching_documents(filters))

    def get_stats(self) -> dict[str, Any]:
        return {"total_documents": len(self.documents)}

    def _matching_documents(self, filters: SearchFilters | None) -> list[IndexDocument]:
        return [
            document
            for document in self.documents.values()
            if _document_matches(document, filters)
        ]


def make_filing(
    ticker: str,
    *,
    accession_number: str,
    filing_date: str,
    section_names: Iterable[str] | None = None,
) -> SimpleNamespace:
    names = list(
        section_names
        or [
            "Item 1. Business",
            "Item 1A. Risk Factors",
            "Item 7. Management's Discussion and Analysis",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        ]
    )
    sections = {
        name: SimpleNamespace(content=_section_text(ticker=ticker, section_name=name))
        for name in names
    }
    return SimpleNamespace(
        metadata=SimpleNamespace(
            ticker=ticker.lower(),
            filing_type="10-K",
            filing_date=filing_date,
            company_name=f"{ticker.upper()} Test Corp.",
            accession_number=accession_number,
            source_url=f"https://www.sec.gov/Archives/{accession_number}",
        ),
        sections=sections,
    )


@pytest.fixture
def sample_filing() -> SimpleNamespace:
    return make_filing(
        "AAPL",
        accession_number="0000320193-25-000001",
        filing_date="2025-09-28",
    )


@pytest.fixture
def representative_filings() -> list[SimpleNamespace]:
    return [
        make_filing("AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28"),
        make_filing("MSFT", accession_number="0000789019-25-000001", filing_date="2025-06-30"),
        make_filing("NVDA", accession_number="0001045810-25-000001", filing_date="2025-01-26"),
    ]


@pytest.fixture
def store() -> InMemoryRetrievalStore:
    return InMemoryRetrievalStore()


def _section_text(*, ticker: str, section_name: str) -> str:
    return (
        f"{ticker.upper()} {section_name} disclosure describes operations, risks, "
        "liquidity, market exposure, controls, customers, products, and suppliers. "
        * 80
    )


def _document_matches(document: IndexDocument, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    metadata = document.metadata
    if filters.ticker and metadata.get("ticker") != filters.ticker.upper():
        return False
    if filters.filing_type and metadata.get("filing_type") != filters.filing_type:
        return False
    if filters.section_key and metadata.get("section_key") != filters.section_key:
        return False
    if filters.section_name and metadata.get("section_name") != filters.section_name:
        return False
    if filters.filing_date and metadata.get("filing_date") != filters.filing_date:
        return False
    return all(metadata.get(key) == value for key, value in filters.extra.items())
