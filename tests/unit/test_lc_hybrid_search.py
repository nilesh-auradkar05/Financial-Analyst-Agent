from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

from app.components.retrieval.hybrid_retrieve import search_langchain_hybrid
from app.components.retrieval.vector_store import (
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)
from tests.unit.fake_retrieval_store import FakeRetrievalStore


def test_langchain_hybrid_uses_bm25_signal_with_store_corpus() -> None:
    result = search_langchain_hybrid(
        FakeRetrievalStore(),
        query="supply chain supplier risks",
        ticker="AAPL",
        filing_type="10-K",
        top_k=2,
    )

    assert result.chunks
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "langchain_hybrid"
    assert result.chunks[0].id == "risk-1"
    assert result.chunks[0].metadata["retrieval_method"] == "langchain_bm25_ensemble_rrf"


def test_langchain_hybrid_supports_test_double_documents() -> None:
    class StoreWithDocuments:
        def __init__(self) -> None:
            self.documents = [
                Document(
                    page_content="Foreign currency exchange and hedging exposures.",
                    metadata={
                        "chunk_id": "market-risk-1",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "section_key": "market_risk",
                    },
                )
            ]

        def search(
            self,
            query: str,
            filters: Optional[SearchFilters] = None,
            n_results: int = 5,
        ) -> SearchResult:
            chunks = [
                RetrievedChunk(
                    id=str(document.metadata["chunk_id"]),
                    text=document.page_content,
                    metadata=dict(document.metadata),
                    distance=0.1,
                )
                for document in self.documents
            ]
            return SearchResult(
                query=query,
                chunks=chunks[:n_results],
                total_results=min(len(chunks), n_results),
                search_time_ms=1.0,
            )

    result = search_langchain_hybrid(
        StoreWithDocuments(),
        query="foreign currency hedging exchange rates",
        ticker="AAPL",
        filing_type="10-K",
        top_k=1,
    )

    assert result.chunks[0].id == "market-risk-1"
