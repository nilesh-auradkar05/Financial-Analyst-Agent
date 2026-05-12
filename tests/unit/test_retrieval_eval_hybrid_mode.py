from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)
from evaluation.retrieval_eval import RetrievalEvalCase, evaluate_retrieval_case


class EvalHybridStubStore:
    def search(self, query, filters=None, n_results=5):
        return SearchResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    id="business-1",
                    text="The company sells products and services.",
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
                        "section_name": "Risk Factors",
                        "section_key": "risk_factors",
                        "chunk_id": "risk-1",
                    },
                    distance=0.4,
                ),
            ],
            total_results=2,
            search_time_ms=4.0,
            filter_used={"ticker": "AAPL"},
        )

    def search_sections(self, ticker, sections, n_results=5, query=None, filing_type=None):
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
        return [
            IndexDocument(
                id="business-1",
                text="The company sells products and services.",
                metadata={"ticker": "AAPL", "section_key": "business"},
            ),
            IndexDocument(
                id="risk-1",
                text="Supply chain disruptions may affect manufacturing and suppliers.",
                metadata={
                    "ticker": "AAPL",
                    "section_name": "Risk Factors",
                    "section_key": "risk_factors",
                },
            ),
        ]


def test_retrieval_eval_supports_hybrid_mode():
    case = RetrievalEvalCase(
        id="aapl-risk-hybrid",
        ticker="AAPL",
        query="What does Apple say about supply chain risks?",
        filing_type="10-K",
        expected_sections=["risk_factors"],
        expected_keywords=["supply chain", "manufacturing"],
        mode="hybrid",
        top_k=2,
    )
    store = EvalHybridStubStore()

    result = evaluate_retrieval_case(case, store=store)

    assert result.passed is True
    assert result.mode == "hybrid"
    assert result.retrieved_sections[0] == "Risk Factors"
    assert result.metrics.first_relevant_rank == 1
    assert result.metrics.mrr_at_5 == 1.0
