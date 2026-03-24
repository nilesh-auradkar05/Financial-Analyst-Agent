from __future__ import annotations

import json
from pathlib import Path

from evaluation.retrieval_eval import (
    RetrievalEvalCase,
    SearchFilters,
    evaluate_retrieval_case,
    load_retrieval_cases,
    summarize_retrieval_results,
)
from rag.vector_store import RetrievedChunk, SearchResult


class StubStore:
    def __init__(self) -> None:
        self.search_called = False
        self.search_sections_called = False

    def search(self, query, filters=None, n_results=5):
        self.search_called = True
        return SearchResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    id="chunk-1",
                    text="General company overview and product portfolio.",
                    metadata={
                        "ticker": "AAPL",
                        "section_name": "business",
                        "chunk_id": "chunk-1",
                    },
                    distance=0.9,
                ),
                RetrievedChunk(
                    id="chunk-2",
                    text="Supply chain disruptions may affect manufacturing and supplier capacity.",
                    metadata={
                        "ticker": "AAPL",
                        "section_name": "risk_factors",
                        "chunk_id": "chunk-2",
                    },
                    distance=0.2,
                ),
            ],
            total_results=2,
            search_time_ms=12.5,
            filter_used={"ticker": "AAPL"},
        )

    def search_sections(self, ticker, sections, n_results=5, query=None, filing_type=None):
        self.search_sections_called = True
        return SearchResult(
            query=query or f"{ticker} sections",
            chunks=[
                RetrievedChunk(
                    id="chunk-3",
                    text="The business section describes core products and competitive dynamics.",
                    metadata={
                        "ticker": ticker,
                        "section_name": sections[0],
                        "chunk_id": "chunk-3",
                    },
                    distance=0.1,
                )
            ],
            total_results=1,
            search_time_ms=7.0,
            filter_used={"ticker": ticker, "sections": sections},
        )


def test_load_retrieval_cases(tmp_path: Path):
    fixture_path = tmp_path / "retrieval_cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "ticker": "aapl",
                    "query": "What are the risks?",
                    "expected_sections": ["risk_factors"],
                }
            ]
        )
    )

    cases = load_retrieval_cases(fixture_path)
    assert len(cases) == 1
    assert cases[0].ticker == "AAPL"
    assert cases[0].expected_sections == ["risk_factors"]


def test_query_mode_reports_section_and_keyword_hits():
    case = RetrievalEvalCase(
        id="aapl-risk",
        ticker="AAPL",
        query="What does Apple say about supply chain risks?",
        expected_sections=["risk_factors"],
        expected_keywords=["supply chain", "supplier"],
        top_k=5,
        mode="query",
    )

    store = StubStore()
    result = evaluate_retrieval_case(case, store=store)

    assert store.search_called is True
    assert result.passed is True
    assert result.metrics.section_hit_at_k is True
    assert result.metrics.section_recall_at_k == 1.0
    assert result.metrics.keyword_hit_rate == 1.0
    assert result.metrics.first_relevant_rank == 2
    assert result.retrieved_sections == ["business", "risk_factors"]


def test_sections_mode_uses_search_sections():
    case = RetrievalEvalCase(
        id="nvda-business",
        ticker="NVDA",
        query="Where does NVIDIA discuss products and competition?",
        expected_sections=["business"],
        expected_keywords=["products"],
        top_k=3,
        mode="sections",
    )

    store = StubStore()
    result = evaluate_retrieval_case(case, store=store)

    assert store.search_sections_called is True
    assert result.passed is True
    assert result.metrics.first_relevant_rank == 1


def test_suite_summary_is_aggregated_cleanly():
    store = StubStore()
    cases = [
        RetrievalEvalCase(
            id="case-1",
            ticker="AAPL",
            query="What are the risks?",
            expected_sections=["risk_factors"],
            expected_keywords=["supply chain"],
        ),
        RetrievalEvalCase(
            id="case-2",
            ticker="NVDA",
            query="Where are the products discussed?",
            expected_sections=["business"],
            expected_keywords=["products"],
            mode="sections",
        ),
    ]

    results = [evaluate_retrieval_case(case, store=store) for case in cases]
    summary = summarize_retrieval_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 1.0
    assert summary["avg_latency_ms"] > 0

    def test_search_filters_backend_filter_with_section_key():
        f = SearchFilters(ticker="aapl", filing_type="10-K", section_key="risk_factors")
        assert f.to_backend_filter() == {
            "$and": [
                {"ticker": "AAPL"},
                {"filing_type": "10-K"},
                {"section_key": "risk_factors"},
            ]
        }

