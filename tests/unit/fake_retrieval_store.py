"""An in-memory stand-in for the Qdrant-backed store used by the reranked-hybrid tests.

It serves the same points through the dense `search` call and the sparse scroll that BM25
reads, so dense and lexical candidates agree. Dense order is the order of `points`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult

BUSINESS_TEXT = "The company sells devices, services, and accessories worldwide."
RISK_TEXT = (
    "Supply chain disruption, supplier concentration, and manufacturing risks may affect production."
)


@dataclass
class FakePoint:
    id: str
    payload: dict[str, object]


def make_point(chunk_id: str, text: str, section_key: str) -> FakePoint:
    return FakePoint(
        id=chunk_id,
        payload={
            "_retrieval_id": chunk_id,
            "text": text,
            "ticker": "AAPL",
            "filing_type": "10-K",
            "section_key": section_key,
        },
    )


DEFAULT_POINTS = (
    make_point("business-1", BUSINESS_TEXT, "business"),
    make_point("risk-1", RISK_TEXT, "risk_factors"),
)


class FakeQdrantClient:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: object,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        offset: Optional[int] = None,
    ):
        del collection_name, scroll_filter, with_payload, with_vectors
        start = int(offset or 0)
        end = min(start + limit, len(self.points))
        next_page = end if end < len(self.points) else None
        return self.points[start:end], next_page


class FakeQdrantStore:
    collection_name = "sec_filings"

    def __init__(self, points: Optional[list[FakePoint]] = None) -> None:
        self.client = FakeQdrantClient(list(points if points is not None else DEFAULT_POINTS))

    def _collection_exists(self) -> bool:
        return True

    def _to_qdrant_filter(self, filters: SearchFilters) -> object:
        return filters.to_backend_filter()

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        del query, filters
        chunks = [
            RetrievedChunk(
                id=point.id,
                text=str(point.payload["text"]),
                metadata={
                    key: value
                    for key, value in point.payload.items()
                    if key in {"ticker", "filing_type", "section_key"}
                },
                distance=0.05 * rank,
            )
            for rank, point in enumerate(self.client.points, start=1)
        ]
        return SearchResult(
            query="dense",
            chunks=chunks[:n_results],
            total_results=min(len(chunks), n_results),
            search_time_ms=1.0,
        )
