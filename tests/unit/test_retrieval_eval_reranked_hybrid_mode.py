from __future__ import annotations

from evaluation.retrieval_eval import RetrievalEvalCase, _search
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult


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


class FakeReranker:
    def predict(self, sentences, *, batch_size=32, show_progress_bar=False):
        del batch_size, show_progress_bar
        return [1.0 for _ in sentences]


def test_retrieval_eval_routes_reranked_hybrid_mode(monkeypatch) -> None:
    import app.components.retrieval.reranked_hybrid_retrieve as module

    monkeypatch.setattr(module, "_get_cross_encoder", lambda model_name, device: FakeReranker())

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
    assert result.filter_used["mode"] in {
        "reranked_hybrid",
        "reranked_hybrid_empty_candidate_fallback",
    }
