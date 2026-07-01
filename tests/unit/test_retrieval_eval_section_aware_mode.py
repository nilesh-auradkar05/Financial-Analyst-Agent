from __future__ import annotations

from app.components.retrieval.vector_store import RetrievedChunk, SearchResult
from evaluation.retrieval_eval import RetrievalEvalCase, evaluate_retrieval_case


class EvalSectionAwareStubStore:
    """Faithful stub for the section_aware dispatch path. section_aware fusion
    should surface the risk_factors chunk for a risk query. Metric *computation*
    is covered by tests/test_retrieval_eval.py; retrieval *quality* by the S2
    content-anchored benchmark."""

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
                )
            ],
            total_results=1,
            search_time_ms=6.0,
            filter_used={"ticker": "AAPL"},
        )

    def search_sections(self, ticker, sections, n_results=5, query=None, filing_type=None):
        return SearchResult(
            query=query or "section query",
            chunks=[
                RetrievedChunk(
                    id="risk-1",
                    text="Supply chain disruptions may affect manufacturing and suppliers.",
                    metadata={
                        "ticker": ticker,
                        "section_name": "Risk Factors",
                        "section_key": "risk_factors",
                        "chunk_id": "risk-1",
                    },
                    distance=0.8,
                )
            ],
            total_results=1,
            search_time_ms=4.0,
            filter_used={"ticker": ticker, "sections": sections},
        )


def test_retrieval_eval_supports_section_aware_mode():
    case = RetrievalEvalCase(
        id="aapl-risk-section-aware",
        ticker="AAPL",
        query="What does Apple say about supply chain risks?",
        filing_type="10-K",
        expected_sections=["risk_factors"],
        expected_keywords=["supply chain", "manufacturing"],
        mode="section_aware",
        top_k=2,
    )
    store = EvalSectionAwareStubStore()

    result = evaluate_retrieval_case(case, store=store)

    # section_aware dispatch must surface the risk_factors chunk for a risk query.
    # We assert observable retrieval output, not which store methods were called.
    assert result.passed is True
    assert "Risk Factors" in result.retrieved_sections
    assert result.metrics.first_relevant_rank == 1
