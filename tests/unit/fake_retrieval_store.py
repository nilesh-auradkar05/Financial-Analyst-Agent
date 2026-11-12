"""An in-memory retrieval store for the hybrid and reranked-hybrid tests.

It serves one set of chunks through the dense `search` call and through the sparse
`iter_sparse_documents` hook that BM25 reads, so dense and lexical candidates agree. Both honour
the metadata filters they are given. Dense order is the order of `points`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

BUSINESS_TEXT = "The company sells devices, services, and accessories worldwide."
RISK_TEXT = (
    "Supply chain disruption, supplier concentration, and manufacturing risks may affect production."
)
_FILTERED_FIELDS = ("ticker", "filing_type", "section_key", "filing_date")


def make_point(chunk_id: str, text: str, section_key: str) -> IndexDocument:
    return IndexDocument(
        id=chunk_id,
        text=text,
        metadata={
            "ticker": "AAPL",
            "filing_type": "10-K",
            "section_key": section_key,
        },
    )


DEFAULT_POINTS = (
    make_point("business-1", BUSINESS_TEXT, "business"),
    make_point("risk-1", RISK_TEXT, "risk_factors"),
)


def _matches(point: IndexDocument, filters: Optional[SearchFilters]) -> bool:
    if filters is None:
        return True
    for name in _FILTERED_FIELDS:
        wanted = getattr(filters, name)
        if wanted is not None and point.metadata.get(name) != wanted:
            return False
    return True


class FakeRetrievalStore:
    def __init__(self, points: Optional[list[IndexDocument]] = None) -> None:
        self.points = list(points if points is not None else DEFAULT_POINTS)

    def iter_sparse_documents(
        self,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None,
    ) -> Iterable[IndexDocument]:
        matching = [point for point in self.points if _matches(point, filters)]
        return matching if limit is None else matching[:limit]

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        del query
        chunks = [
            RetrievedChunk(
                id=point.id,
                text=point.text,
                metadata=dict(point.metadata),
                distance=0.05 * rank,
            )
            for rank, point in enumerate(
                (point for point in self.points if _matches(point, filters)), start=1
            )
        ]
        return SearchResult(
            query="dense",
            chunks=chunks[:n_results],
            total_results=min(len(chunks), n_results),
            search_time_ms=1.0,
        )
