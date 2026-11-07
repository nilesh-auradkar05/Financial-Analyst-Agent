from __future__ import annotations

from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult
from evaluation.retrieval_eval import RetrievalEvalCase, _search
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, RERANK_PATH


class FakeStore:
    def search(self, query: str, filters: SearchFilters | None = None, n_results: int = 5) -> SearchResult:
        del query, filters, n_results
        return SearchResult(
            query="dense",
            chunks=[
                RetrievedChunk(
                    id="risk-1",
                    text="Cybersecurity and data security incidents could harm operations.",
                    metadata={"ticker": "MSFT", "filing_type": "10-K", "section_key": "risk_factors"},
                    distance=0.1,
                )
            ],
            total_results=1,
            search_time_ms=1.0,
        )


def test_retrieval_eval_routes_reranked_hybrid_mode(foundry_rerank) -> None:
    """test-plan §25: Rerank reorders candidates (the eval mode reranks on the hosted provider)."""
    case = RetrievalEvalCase(
        id="reranked_hybrid_eval",
        ticker="MSFT",
        query="cybersecurity data security risks",
        filing_type="10-K",
        expected_sections=["risk_factors"],
        expected_keywords=["cybersecurity"],
        mode="reranked_hybrid",
    )

    result = _search(FakeStore(), case)

    assert result.has_results
    assert result.filter_used is not None
    assert result.filter_used["mode"] == "reranked_hybrid"
    assert result.chunks[0].metadata["reranker_model"] == f"foundry:{RERANK_DEPLOYMENT}"
    (request,) = foundry_rerank.requests
    assert request.path == RERANK_PATH
    assert request.body["query"] == case.query
