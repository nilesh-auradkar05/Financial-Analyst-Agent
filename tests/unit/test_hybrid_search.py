from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from rag.hybrid_search import HybridSearchConfig, search_hybrid
from rag.vector_store import IndexDocument, RetrievedChunk, SearchFilters, SearchResult


class HybridStubStore:
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
                    distance=0.05,
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
                    distance=0.30,
                ),
            ],
            total_results=2,
            search_time_ms=5.0,
            filter_used={"ticker": "AAPL"},
        )

    def search_sections(self, ticker, sections, n_results=5, query=None, filing_type=None):
        self.search_sections_called = True
        return SearchResult(
            query=query or "section query",
            chunks=[],
            total_results=0,
            search_time_ms=1.0,
            filter_used={"ticker": ticker, "sections": sections},
        )

    def iter_sparse_documents(
        self,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None,
    ) -> Iterable[IndexDocument]:
        documents = [
            IndexDocument(
                id="business-1",
                text="The company sells phones, computers, and services.",
                metadata={"ticker": "AAPL", "section_key": "business"},
            ),
            IndexDocument(
                id="risk-1",
                text="Supply chain disruptions may affect manufacturing and suppliers.",
                metadata={"ticker": "AAPL", "section_key": "risk_factors"},
            ),
        ]
        return documents[: limit or len(documents)]


def test_hybrid_search_promotes_sparse_supported_result():
    store = HybridStubStore()

    result = search_hybrid(
        store,
        query="What does Apple say about supply chain risks?",
        ticker="AAPL",
        filing_type="10-K",
        top_k=2,
        config=HybridSearchConfig(
            dense_candidate_k=2,
            sparse_candidate_k=2,
            final_top_k=2,
            sparse_weight=2.0,
            use_section_aware_dense=False,
        ),
    )

    assert store.search_called is True
    assert result.chunks[0].id == "risk-1"
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "hybrid"
    assert "hybrid_score" in result.chunks[0].metadata
