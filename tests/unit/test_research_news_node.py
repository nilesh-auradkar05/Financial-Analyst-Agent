from __future__ import annotations

import pytest

from app.agents import graph as graph_mod
from app.agents.state import AgentStep, create_initial_state


@pytest.mark.asyncio
async def test_research_news_node_skips_search_when_max_news_articles_is_zero(monkeypatch):
    async def fail_search_company_news(query: str, max_results: int = 5):
        raise AssertionError("search_company_news should not be called")

    monkeypatch.setattr(graph_mod, "search_company_news", fail_search_company_news)

    state = create_initial_state("AAPL", max_news_articles=0)
    result = await graph_mod.research_news_node(state)

    assert result == {
        "news_articles": [],
        "current_step": AgentStep.RESEARCH_NEWS.value,
    }
