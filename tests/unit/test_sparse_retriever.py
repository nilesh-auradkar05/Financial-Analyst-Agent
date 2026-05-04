from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from rag.sparse_retriever import search_sparse
from rag.vector_store import IndexDocument, SearchFilters


class SparseDocumentStubStore:
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
            IndexDocument(
                id="market-risk-1",
                text="Foreign currency exchange rates may affect financial results.",
                metadata={"ticker": "AAPL", "section_key": "market_risk"},
            ),
        ]
        return documents[: limit or len(documents)]


def test_sparse_search_ranks_keyword_matching_chunk_first():
    store = SparseDocumentStubStore()

    result = search_sparse(
        store,
        query="What does Apple say about supply chain risks?",
        filters=SearchFilters(ticker="AAPL", filing_type="10-K"),
        top_k=3,
    )

    assert result.chunks[0].id == "risk-1"
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "sparse"
    assert result.chunks[0].metadata["sparse_rank"] == 1
    assert result.chunks[0].metadata["sparse_score"] > 0.0
