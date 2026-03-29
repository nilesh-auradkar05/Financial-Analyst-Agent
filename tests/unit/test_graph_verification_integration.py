from __future__ import annotations

import pytest

from agents.graph import draft_memo_node, verify_memo_node
from agents.state import create_initial_state


class StubLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class StubLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages):
        return StubLLMResponse(self._content)


@pytest.mark.asyncio
async def test_draft_and_verify_nodes_attach_passed_verification(monkeypatch):
    memo = (
        "# Investment Memo\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $394 billion [1]. "
        "Supply chain concentration in China remains a risk [2]."
    )

    monkeypatch.setattr("agents.graph.get_llm", lambda temperature=0.7: StubLLM(memo))

    state = create_initial_state("AAPL", "Apple Inc.")
    state["news_articles"] = [
        {
            "title": "Apple annual revenue update",
            "source": "Example News",
            "snippet": "Apple reported revenue of $394 billion in fiscal 2024.",
            "published_date": "2026-03-15",
            "url": "https://example.com/apple-revenue",
        }
    ]
    state["filing_chunks"] = [
        {
            "chunk_id": "chunk-1",
            "section": "Risk Factors",
            "filing_type": "10-K",
            "filing_date": "2025-09-28",
            "text": "Risks include supply chain concentration in China and dependence on manufacturing partners.",
        }
    ]

    draft_update = await draft_memo_node(state)
    merged_state = {**state, **draft_update}
    verify_update = await verify_memo_node(merged_state)

    assert draft_update["citations"]
    assert draft_update["citation_evidence"]
    assert verify_update["verification_result"]["passed"] is True
    assert verify_update["verification_result"]["citation_coverage_rate"] == 1.0
    assert verify_update["verification_result"]["grounded_claim_rate"] == 1.0
    assert verify_update["verification_result"]["orphan_citations"] == []


@pytest.mark.asyncio
async def test_verify_memo_flags_orphan_and_numeric_mismatch(monkeypatch):
    memo = (
        "# Investment Memo\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $500 billion [1]. "
        "A second unsupported statement cites [99]."
    )

    monkeypatch.setattr("agents.graph.get_llm", lambda temperature=0.7: StubLLM(memo))

    state = create_initial_state("AAPL", "Apple Inc.")
    state["news_articles"] = [
        {
            "title": "Apple annual revenue update",
            "source": "Example News",
            "snippet": "Apple reported revenue of $394 billion in fiscal 2024.",
            "published_date": "2026-03-15",
            "url": "https://example.com/apple-revenue",
        }
    ]

    draft_update = await draft_memo_node(state)
    merged_state = {**state, **draft_update}
    verify_update = await verify_memo_node(merged_state)

    assert verify_update["verification_result"]["passed"] is False
    assert verify_update["verification_result"]["orphan_citations"] == [99]
    assert any(error["step"] == "verify_memo" for error in verify_update["errors"])
