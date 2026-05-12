from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from app.components.retrieval.hybrid_retrieve import search_langchain_hybrid
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult


@dataclass
class FakePoint:
    id: str
    payload: dict[str, object]


class FakeQdrantClient:
    def __init__(self, points: list[FakePoint]):
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

    def __init__(self) -> None:
        self.client = FakeQdrantClient(
            [
                FakePoint(
                    id="risk-1",
                    payload={
                        "_retrieval_id": "risk-1",
                        "text": "Supply chain disruption and supplier concentration risks may affect production.",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "section_key": "risk_factors",
                    },
                ),
                FakePoint(
                    id="business-1",
                    payload={
                        "_retrieval_id": "business-1",
                        "text": "The company sells phones, services, and accessories worldwide.",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "section_key": "business",
                    },
                ),
            ]
        )

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
                id="business-1",
                text="The company sells phones, services, and accessories worldwide.",
                metadata={"ticker": "AAPL", "filing_type": "10-K", "section_key": "business"},
                distance=0.1,
            ),
            RetrievedChunk(
                id="risk-1",
                text="Supply chain disruption and supplier concentration risks may affect production.",
                metadata={"ticker": "AAPL", "filing_type": "10-K", "section_key": "risk_factors"},
                distance=0.2,
            ),
        ]
        return SearchResult(
            query="dense",
            chunks=chunks[:n_results],
            total_results=min(len(chunks), n_results),
            search_time_ms=1.0,
        )


def test_langchain_hybrid_uses_bm25_signal_with_qdrant_corpus() -> None:
    result = search_langchain_hybrid(
        FakeQdrantStore(),
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
    class StoreWithDocuments(FakeQdrantStore):
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

        @property
        def client(self):  # type: ignore[override]
            raise AttributeError("not qdrant")

    result = search_langchain_hybrid(
        StoreWithDocuments(),
        query="foreign currency hedging exchange rates",
        ticker="AAPL",
        filing_type="10-K",
        top_k=1,
    )

    assert result.chunks[0].id == "market-risk-1"
