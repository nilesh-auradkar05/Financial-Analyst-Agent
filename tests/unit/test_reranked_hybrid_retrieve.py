from __future__ import annotations

import re

from app.components.retrieval.reranked_hybrid_retrieve import (
    RerankedHybridSearchConfig,
    search_reranked_hybrid,
)
from tests.unit.fake_retrieval_store import FakeQdrantStore


class OverlapReranker:
    """A computed scorer at the provider boundary: the share of query words a document contains."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        wanted = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [
            len(wanted & set(re.findall(r"[a-z0-9]+", document.lower()))) / len(wanted)
            for document in documents
        ]


def test_reranked_hybrid_promotes_reranker_preferred_candidate() -> None:
    """test-plan §25: Rerank reorders candidates."""
    result = search_reranked_hybrid(
        FakeQdrantStore(),
        query="supply chain supplier risks",
        ticker="AAPL",
        filing_type="10-K",
        top_k=2,
        reranker=OverlapReranker(),
        config=RerankedHybridSearchConfig(dense_top_k=2, sparse_top_k=2, final_top_k=2),
    )

    assert result.chunks
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "reranked_hybrid"
    assert result.chunks[0].id == "risk-1"
    assert result.chunks[0].metadata["retrieval_method"] == "dense_bm25_cross_encoder_rerank"
    assert result.chunks[0].metadata["reranker_score"] == 1.0
    assert result.chunks[0].metadata["dense_rank"] == 2
    assert result.chunks[0].metadata["bm25_rank"] is not None


def test_reranked_hybrid_reports_candidate_count() -> None:
    """test-plan §25: Rerank reorders candidates (a supplied reranker scores every candidate)."""
    result = search_reranked_hybrid(
        FakeQdrantStore(),
        query="devices services business",
        ticker="AAPL",
        filing_type="10-K",
        top_k=1,
        reranker=OverlapReranker(),
        config=RerankedHybridSearchConfig(dense_top_k=2, sparse_top_k=2, final_top_k=1),
    )

    assert result.filter_used is not None
    assert result.filter_used["candidate_count"] >= 1
