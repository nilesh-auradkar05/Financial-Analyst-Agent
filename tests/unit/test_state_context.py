from __future__ import annotations

from agents.state import create_initial_state, get_context_for_llm


def test_get_context_for_llm_does_not_number_sources():
    state = create_initial_state("AAPL", "Apple Inc.")
    state["news_articles"] = [
        {
            "title": "Apple launches new product",
            "source": "Example News",
            "snippet": "The company introduced a new product line.",
        }
    ]
    state["filing_chunks"] = [
        {
            "section": "Risk Factors",
            "text": "Supply chain concentration remains a key risk.",
        }
    ]

    context = get_context_for_llm(state)

    assert "[1]" not in context
    assert "[2]" not in context
    assert "- Apple launches new product" in context
    assert "- Risk Factors" in context
