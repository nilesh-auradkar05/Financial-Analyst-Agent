from __future__ import annotations

from app.components.retrieval.vector_store import RetrievedChunk, SearchResult
from evaluation.retrieval_eval import RetrievalEvalCase, evaluate_retrieval_case


class EvalSectionAwareStubStore:
    def __init__(self) -> None:
        self.search_called = False
        self.search_sections_called = False

    def search(self, query, filters=None, n_results=5):
        self.search_called = True
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
        self.search_sections_called = True
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

    assert store.search_called is True
    assert store.search_sections_called is True
    assert result.passed is True
    assert result.retrieved_sections[0] == "Risk Factors"
    assert result.metrics.first_relevant_rank == 1
    assert result.metrics.mrr_at_5 == 1.0
