from __future__ import annotations

from app.components.retrieval.section_aware_search import search_section_aware
from app.components.retrieval.vector_store import RetrievedChunk, SearchResult


class SectionAwareStubStore:
    def __init__(self) -> None:
        self.search_called = False
        self.search_sections_called = False

    def search(self, query, filters=None, n_results=5):
        self.search_called = True
        return SearchResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    id="business-1",
                    text="The company sells phones, computers, and services.",
                    metadata={
                        "ticker": "AAPL",
                        "section_name": "business",
                        "section_key": "business",
                        "chunk_id": "business-1",
                    },
                    distance=0.1,
                ),
                RetrievedChunk(
                    id="risk-1",
                    text="Supply chain disruptions may affect manufacturing and suppliers.",
                    metadata={
                        "ticker": "AAPL",
                        "section_name": "risk_factors",
                        "section_key": "risk_factors",
                        "chunk_id": "risk-1",
                    },
                    distance=0.7,
                ),
            ],
            total_results=2,
            search_time_ms=8.0,
            filter_used={"ticker": "AAPL"},
        )

    def search_sections(self, ticker, sections, n_results=5, query=None, filing_type=None):
        self.search_sections_called = True
        return SearchResult(
            query=query or "section query",
            chunks=[
                RetrievedChunk(
                    id="risk-1",
                    text="Supply chain disruptions may affect manufacturing and suppliers.",
                    metadata={
                        "ticker": ticker,
                        "section_name": "Risk Factors",
                        "section_key": "risk_factors",
                        "chunk_id": "risk-1",
                    },
                    distance=0.7,
                )
            ],
            total_results=1,
            search_time_ms=5.0,
            filter_used={"ticker": ticker, "sections": sections},
        )


def test_section_aware_search_promotes_inferred_section_result():
    store = SectionAwareStubStore()

    result = search_section_aware(
        store,
        query="What does Apple say about supply chain risks?",
        ticker="AAPL",
        filing_type="10-K",
        top_k=2,
    )

    assert store.search_called is True
    assert store.search_sections_called is True
    assert result.chunks[0].id == "risk-1"
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "section_aware"
    assert result.filter_used["target_sections"] == ["risk_factors"]
    assert "section_aware_score" in result.chunks[0].metadata
