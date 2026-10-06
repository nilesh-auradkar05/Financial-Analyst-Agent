"""Trace: docs/test-plan.md §16, analysis without filings and retrieval failure."""

from types import SimpleNamespace

import pytest

from app.agents import graph
from app.agents.state import apply_update, create_initial_state


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval_fails", [False, True])
async def test_missing_filing_keeps_available_evidence_and_verifies(monkeypatch, retrieval_fails):
    state = create_initial_state("AAPL", "Apple Inc.")
    state["stock_data"] = {"ticker": "AAPL", "current_price": 123.0}
    state["news_articles"] = [{"title": "Market update", "snippet": "Shares rose.", "source": "Wire"}]
    state["sentiment_result"] = {"overall_sentiment": "positive", "positive_count": 1}

    monkeypatch.setattr(graph, "get_vector_store", lambda: object())
    def search(*_args):
        if retrieval_fails:
            raise RuntimeError("secret retrieval details")
        return SimpleNamespace(chunks=[])
    monkeypatch.setattr(graph, "_search_filing_chunks", search)

    state = apply_update(state, await graph.retrieve_sec_filings_node(state))

    class MemoLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="# Investment Memo\n\nMarket evidence is limited.")

    monkeypatch.setattr(graph, "get_llm", lambda _settings: MemoLLM())
    state = apply_update(state, await graph.draft_memo_node(state))
    verification = await graph.verify_memo_node(state)

    assert state["stock_data"]["current_price"] == 123.0
    assert len(state["news_articles"]) == 1
    assert state["sentiment_result"]["overall_sentiment"] == "positive"
    assert state.get("filing_chunks") == []
    assert "SEC Filings: Not Available" in state.get("investment_memo", "")
    assert "AAPL" in state.get("investment_memo", "")
    assert all(item["source_type"] != "sec_filing" for item in state.get("citations", []))
    assert "verification_result" in verification
    reason = next(error["message"] for error in state.get("errors", []) if error["step"] == "retrieve_filings")
    assert reason == ("Filing retrieval failed for AAPL." if retrieval_fails else "No indexed filings found for AAPL.")


@pytest.mark.asyncio
async def test_disabled_filing_analysis_has_no_missing_warning(monkeypatch):
    state = create_initial_state("AAPL", include_filing_analysis=False)
    retrieval = await graph.retrieve_sec_filings_node(state)
    state.update(retrieval)

    class MemoLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="# Memo\n\nNo conclusion.")

    monkeypatch.setattr(graph, "get_llm", lambda _settings: MemoLLM())
    draft = await graph.draft_memo_node(state)
    assert "SEC Filings: Not Available" not in draft["investment_memo"]
    assert not any(error["step"] == "retrieve_filings" for error in draft["errors"])
