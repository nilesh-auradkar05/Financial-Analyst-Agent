from __future__ import annotations

from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult
from evaluation.retrieval_eval import RetrievalEvalCase, _search


class FakeStore:
    documents: list[RetrievedChunk] = [
        RetrievedChunk(
            id="risk-1",
            text="Cybersecurity and data security incidents could harm operations.",
            metadata={"ticker": "MSFT", "filing_type": "10-K", "section_key": "risk_factors"},
            distance=0.1,
        )
    ]

    def search(self, query: str, filters: SearchFilters | None = None, n_results: int = 5) -> SearchResult:
        del query, filters, n_results
        return SearchResult(
            query="dense",
            chunks=self.documents,
            total_results=1,
            search_time_ms=1.0,
        )


def test_retrieval_eval_routes_hybrid_mode_to_langchain_hybrid() -> None:
    case = RetrievalEvalCase(
        id="hybrid_eval",
        ticker="MSFT",
        query="cybersecurity data security risks",
        filing_type="10-K",
        expected_sections=["risk_factors"],
        expected_keywords=["cybersecurity"],
        mode="hybrid",
    )

    result = _search(FakeStore(), case)

    assert result.has_results
    assert result.filter_used is not None
    assert result.filter_used["mode"] in {
        "langchain_hybrid",
        "langchain_hybrid_dense_fallback",
    }
