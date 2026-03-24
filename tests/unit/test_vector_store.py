from __future__ import annotations

import rag.vector_store as vector_store
from rag.vector_store import (
    ChromaDBVectorStore,
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)


def _make_chunk(chunk_id: str, section_name: str = "Risk Factors") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"Chunk for {section_name}",
        metadata={
            "ticker": "AAPL",
            "section": section_name,
            "section_name": section_name,
            "section_key": "risk_factors",
            "filing_date": "2025-09-28",
        },
        distance=0.1,
    )

def test_canonical_section_key_maps_common_variants():
    assert vector_store.canonical_section_key("Business") == "business"
    assert vector_store.canonical_section_key("Item 1A. Risk Factors") == "risk_factors"
    assert vector_store.canonical_section_key("Management's Discussion and Analysis") == "md&a"
    assert (
        vector_store.canonical_section_key(
            "Item 7A Quantitative and Qualitative Disclosures About Market Risk"
        )
        == "market_risk"
    )

def test_search_filters_prefer_section_key_when_present():
    filters = SearchFilters(
        ticker="aapl",
        filing_type="10-K",
        section_name="Risk Factors",
        section_key="risk_factors",
        filing_date="2025-09-28",
    )

    assert filters.to_backend_filter() == {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section_key": "risk_factors",
        "filing_date": "2025-09-28",
    }

def test_search_sections_prefers_canonical_section_key(monkeypatch):
    store = object.__new__(vector_store.ChromaDBVectorStore)
    calls: list[SearchFilters] = []

    def fake_search(*, query: str, filters: SearchFilters | None = None, n_results: int = 5):
        calls.append(filters)
        return SearchResult(
            query=query,
            chunks=[_make_chunk("risk-001")],
            total_results=1,
            search_time_ms=1.5,
            filter_used=filters.to_backend_filter() if filters else None,
        )

    monkeypatch.setattr(store, "search", fake_search)

    result = vector_store.ChromaDBVectorStore.search_sections(
        store,
        ticker="AAPL",
        sections=["Risk Factors"],
        n_results=3,
        query="supply chain exposure",
        filing_type="10-K",
    )

    assert result.total_results == 1
    assert len(calls) == 1
    assert calls[0].ticker == "AAPL"
    assert calls[0].filing_type == "10-K"
    assert calls[0].section_key == "risk_factors"
    assert calls[0].section_name is None

def test_search_sections_fall_back_to_legacy_section_name(monkeypatch):
    store = object.__new__(vector_store.ChromaDBVectorStore)
    calls: list[SearchFilters] = []

    def fake_search(*, query: str, filters: SearchFilters | None = None, n_results: int = 5):
        calls.append(filters)

        if filters and filters.section_key == "risk_factors":
            return SearchResult(
                query=query,
                chunks=[],
                total_results=0,
                search_time_ms=1.0,
                filter_used=filters.to_backend_filter(),
            )

        if filters and filters.section_name == "Risk Factors":
            return SearchResult(
                query=query,
                chunks=[_make_chunk("risk-legacy-001")],
                total_results=1,
                search_time_ms=1.0,
                filter_used=filters.to_backend_filter(),
            )

        return SearchResult(
            query=query,
            chunks=[],
            total_results=0,
            search_time_ms=1.0,
            filter_used=filters.to_backend_filter() if filters else None,
        )

    monkeypatch.setattr(store, "search", fake_search)

    result = vector_store.ChromaDBVectorStore.search_sections(
        store,
        ticker="AAPL",
        sections=["Risk Factors"],
        n_results=3,
        query="supplier concentration",
        filing_type="10-K",
    )

    assert result.total_results == 1
    assert [call.section_key for call in calls] == ["risk_factors", None]
    assert calls[1].section_name == "Risk Factors"

def test_search_by_ticker_uses_section_fallback_path(monkeypatch):
    store = object.__new__(vector_store.ChromaDBVectorStore)
    calls: list[SearchFilters] = []

    def fake_search(*, query: str, filters: SearchFilters | None = None, n_results: int = 5):
        calls.append(filters)
        return SearchResult(
            query=query,
            chunks=[_make_chunk("business-001", section_name="Business")],
            total_results=1,
            search_time_ms=0.5,
            filter_used=filters.to_backend_filter() if filters else None,
        )

    monkeypatch.setattr(store, "search", fake_search)

    result = vector_store.ChromaDBVectorStore.search_by_ticker(
        store,
        query="core products",
        ticker="AAPL",
        n_results=2,
        section="Business",
    )

    assert result.total_results == 1
    assert calls[0].section_key == "business"
    assert calls[0].ticker == "AAPL"

def test_add_documents_strips_none_metadata_values():
    store = ChromaDBVectorStore(collection_name="test_sec_filings", persist_directory=None)

    docs = [
        IndexDocument(
            id="doc1",
            text="sample text",
            metadata={
                "ticker": "AAPL",
                "section": "Risk Factors",
                "company_name": None,
                "source_url": None,
                "chunk_index": 0,
            },
        )
    ]

    normalized = store._normalize_documents(
        documents=docs,
        texts=None,
        metadatas=None,
        ids=None,
    )

    assert normalized[0].metadata == {
        "ticker": "AAPL",
        "section": "Risk Factors",
        "chunk_index": 0,
    }
