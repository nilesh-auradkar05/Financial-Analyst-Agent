from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from rag.reranked_hybrid_retrieve import (
    RerankedHybridSearchConfig,
    search_reranked_hybrid,
)
from rag.vector_store import RetrievedChunk, SearchFilters, SearchResult


@dataclass
class FakePoint:
    id: str
    payload: dict[str, object]


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

    def __init__(self) -> None:
        self.client = FakeQdrantClient(
            [
                FakePoint(
                    id="business-1",
                    payload={
                        "_retrieval_id": "business-1",
                        "text": "The company sells devices, services, and accessories worldwide.",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "section_key": "business",
                    },
                ),
                FakePoint(
                    id="risk-1",
                    payload={
                        "_retrieval_id": "risk-1",
                        "text": "Supply chain disruption, supplier concentration, and manufacturing risks may affect production.",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "section_key": "risk_factors",
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
                text="The company sells devices, services, and accessories worldwide.",
                metadata={"ticker": "AAPL", "filing_type": "10-K", "section_key": "business"},
                distance=0.05,
            ),
            RetrievedChunk(
                id="risk-1",
                text="Supply chain disruption, supplier concentration, and manufacturing risks may affect production.",
                metadata={"ticker": "AAPL", "filing_type": "10-K", "section_key": "risk_factors"},
                distance=0.10,
            ),
        ]
        return SearchResult(
            query="dense",
            chunks=chunks[:n_results],
            total_results=min(len(chunks), n_results),
            search_time_ms=1.0,
        )


class KeywordFakeReranker:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        del batch_size, show_progress_bar
        scores: list[float] = []
        for _query, document in sentences:
            text = document.lower()
            scores.append(10.0 if "supply chain" in text else 1.0)
        return scores


def test_reranked_hybrid_promotes_cross_encoder_preferred_candidate() -> None:
    result = search_reranked_hybrid(
        FakeQdrantStore(),
        query="supply chain supplier risks",
        ticker="AAPL",
        filing_type="10-K",
        top_k=2,
        reranker=KeywordFakeReranker(),
        config=RerankedHybridSearchConfig(
            dense_top_k=2,
            sparse_top_k=2,
            final_top_k=2,
            reranker_model_name="fake-reranker",
        ),
    )

    assert result.chunks
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "reranked_hybrid"
    assert result.chunks[0].id == "risk-1"
    assert result.chunks[0].metadata["retrieval_method"] == "dense_bm25_cross_encoder_rerank"
    assert result.chunks[0].metadata["reranker_score"] == 10.0
    assert result.chunks[0].metadata["dense_rank"] == 2
    assert result.chunks[0].metadata["bm25_rank"] is not None


def test_reranked_hybrid_reports_candidate_count() -> None:
    result = search_reranked_hybrid(
        FakeQdrantStore(),
        query="devices services business",
        ticker="AAPL",
        filing_type="10-K",
        top_k=1,
        reranker=KeywordFakeReranker(),
        config=RerankedHybridSearchConfig(
            dense_top_k=2,
            sparse_top_k=2,
            final_top_k=1,
            reranker_model_name="fake-reranker",
        ),
    )

    assert result.filter_used is not None
    assert result.filter_used["candidate_count"] >= 1
