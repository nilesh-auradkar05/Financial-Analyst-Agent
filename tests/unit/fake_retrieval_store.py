"""An in-memory retrieval store for the reranked-retrieval tests (test-plan §25).

It serves one set of chunks through two ranked calls, each of which honours the metadata filters it
is given and returns at most `n_results`:

- `search` is the dense stand-in: the order of `points`.
- `search_hybrid` is the fused stand-in: character-trigram similarity between the query and the
  chunk text, best first, ties in the order of `points`. It ranks differently from `search`, so a
  test can tell which of the two a caller used.
"""

from __future__ import annotations

from typing import Optional

from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

BUSINESS_TEXT = "The company sells devices, services, and accessories worldwide."
RISK_TEXT = "Supply chain disruption, supplier concentration, and manufacturing risks may affect production."
_FILTERED_FIELDS = ("ticker", "filing_type", "section_key", "filing_date")


def make_point(
    chunk_id: str,
    text: str,
    section_key: str,
    ticker: str = "AAPL",
    filing_type: str = "10-K",
) -> IndexDocument:
    return IndexDocument(
        id=chunk_id,
        text=text,
        metadata={
            "ticker": ticker,
            "filing_type": filing_type,
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


def _trigrams(text: str) -> set[str]:
    padded = (
        f"  {' '.join(sorted(set(text.lower().replace(',', ' ').replace('.', ' ').split())))}  "
    )
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _similarity(query: str, text: str) -> float:
    wanted, found = _trigrams(query), _trigrams(text)
    return len(wanted & found) / len(wanted | found)


class FakeRetrievalStore:
    def __init__(self, points: Optional[list[IndexDocument]] = None) -> None:
        self.points = list(points if points is not None else DEFAULT_POINTS)

    def _result(
        self, query: str, points: list[IndexDocument], distances: list[float], n_results: int
    ) -> SearchResult:
        chunks = [
            RetrievedChunk(
                id=point.id,
                text=point.text,
                metadata=dict(point.metadata),
                distance=distance,
            )
            for point, distance in zip(points, distances)
        ][:n_results]
        return SearchResult(
            query=query,
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=1.0,
        )

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        matching = [point for point in self.points if _matches(point, filters)]
        distances = [0.05 * rank for rank in range(1, len(matching) + 1)]
        return self._result(query, matching, distances, n_results)

    def search_hybrid(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        matching = [point for point in self.points if _matches(point, filters)]
        ranked = sorted(matching, key=lambda point: -_similarity(query, point.text))
        distances = [1.0 - _similarity(query, point.text) for point in ranked]
        return self._result(query, ranked, distances, n_results)
