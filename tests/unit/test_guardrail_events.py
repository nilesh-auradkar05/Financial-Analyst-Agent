"""Trace: docs/test-plan.md section 12, S7.1 rows "Source allow-list" (the node that applies it),
"Untrusted-content framing" (`untrusted_content_framed`) and "Guardrail events returned".

Real nodes and the real compiled graph run over the computing tool fakes of the fan-out suite
(tests/unit/test_graph_fanout.py); the news source and the memo provider are the boundary fakes.
New modules and symbols are imported inside the tests so each test fails on its own until they
exist.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agents import graph
from app.agents.graph import run_agent
from app.agents.state import AgentState, create_initial_state
from tests.unit.guardrail_support import CapturingLLM, evidence_state
from tests.unit.test_graph_fanout import EchoLLM, install_fakes

FRAMED = {"code": "untrusted_content_framed", "detail": None}


@pytest.fixture(autouse=True)
def blank_search_key(monkeypatch):
    """Keeps the developer's real search key out of any assertion or error message."""
    monkeypatch.setattr(graph.settings.tavily, "api_key", "test-key")


def article(url: str, text: str = "Apple revenue rose in the quarter.") -> SimpleNamespace:
    return SimpleNamespace(
        title=f"Story at {url}",
        url=url,
        source="Wire",
        snippet=text,
        published_date="2026-09-01",
    )


def install_news(monkeypatch, *items: SimpleNamespace) -> None:
    async def news(query, max_results=10):
        return list(items)

    monkeypatch.setattr(graph, "search_company_news", news)


# --- the state ------------------------------------------------------------------------------------


def test_a_new_state_has_no_guardrail_events():
    """12 'Guardrail events returned': `guardrail_events` is part of the state, empty at the start."""
    assert create_initial_state("AAPL")["guardrail_events"] == []


async def test_events_from_parallel_nodes_are_all_kept_and_earlier_ones_are_not_overwritten():
    """12 'Guardrail events returned': the state appends events, so two nodes that run in the same
    step both keep theirs and a later node does not replace what an earlier one recorded."""

    async def left(state):
        return {"guardrail_events": [{"code": "source_dropped", "detail": "left.example"}]}

    async def right(state):
        return {
            "guardrail_events": [
                {"code": "source_dropped", "detail": "right.example"},
                {"code": "untrusted_content_framed", "detail": None},
            ]
        }

    async def after(state):
        return {"guardrail_events": [{"code": "pii_redacted", "detail": None}]}

    builder = StateGraph(AgentState)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("after", after)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", "after")
    builder.add_edge("right", "after")
    builder.add_edge("after", END)

    final = await builder.compile().ainvoke(create_initial_state("AAPL"))

    recorded = [(event["code"], event["detail"]) for event in final["guardrail_events"]]
    assert sorted(recorded, key=str) == sorted(
        [
            ("source_dropped", "left.example"),
            ("source_dropped", "right.example"),
            ("untrusted_content_framed", None),
            ("pii_redacted", None),
        ],
        key=str,
    )
    assert recorded[-1] == ("pii_redacted", None)


# --- the response model ---------------------------------------------------------------------------


def test_a_guardrail_event_response_has_a_code_and_an_optional_detail_and_nothing_else():
    """12 'Guardrail events returned': `{code, detail}`; extra fields are refused."""
    from app.models import GuardrailEventResponse

    assert GuardrailEventResponse(code="mnpi_removed").detail is None
    assert GuardrailEventResponse(code="source_dropped", detail="evil.example").model_dump() == {
        "code": "source_dropped",
        "detail": "evil.example",
    }
    with pytest.raises(ValidationError):
        GuardrailEventResponse(code="mnpi_removed", detail=None, sentence="the removed text")


def test_the_response_lists_no_events_by_default_and_does_not_share_the_default():
    """12 'Guardrail events returned': defaults to []; the default list is not shared."""
    from app.models import AnalysisResponse, GuardrailEventResponse, JobStatus

    first = AnalysisResponse(ticker="AAPL", company_name="Apple Inc.", status=JobStatus.COMPLETED)
    second = AnalysisResponse(
        ticker="MSFT", company_name="Microsoft Corp.", status=JobStatus.COMPLETED
    )

    first.guardrail_events.append(GuardrailEventResponse(code="mnpi_removed"))

    assert second.guardrail_events == []


def test_the_response_passes_the_states_events_through_in_order():
    """12 'Guardrail events returned': the response is built from the state's events."""
    from app.main import _format_response

    events = [
        {"code": "source_dropped", "detail": "evil.example"},
        {"code": "untrusted_content_framed", "detail": None},
    ]
    state = {"ticker": "AAPL", "company_name": "Apple Inc.", "guardrail_events": events}

    response = _format_response(state)

    assert [event.model_dump() for event in response.guardrail_events] == events
    assert _format_response({"ticker": "AAPL", "company_name": "Apple Inc."}).guardrail_events == []


# --- the news node --------------------------------------------------------------------------------


async def test_the_news_node_keeps_allowed_articles_and_reports_each_drop(monkeypatch):
    """12 'Source allow-list': the live news node keeps allowed hosts (subdomains too), drops the
    rest, and records source_dropped with the host as detail."""
    install_news(
        monkeypatch,
        article("https://evil.example/apple"),
        article("https://www.reuters.com/apple"),
        article("https://reuters.com.evil.example/apple"),
        article("https://finance.yahoo.com/quote/AAPL"),
        article("ftp://www.reuters.com/apple"),
    )

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    assert [item["url"] for item in update["news_articles"]] == [
        "https://www.reuters.com/apple",
        "https://finance.yahoo.com/quote/AAPL",
    ]
    assert [event["code"] for event in update["guardrail_events"]] == ["source_dropped"] * 3
    assert [event["detail"] for event in update["guardrail_events"]][:2] == [
        "evil.example",
        "reuters.com.evil.example",
    ]


async def test_the_news_node_records_nothing_when_nothing_is_dropped(monkeypatch):
    """12 'Guardrail events returned': a run that triggers nothing records only what fired."""
    install_news(monkeypatch, article("https://www.reuters.com/apple"))

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    assert len(update["news_articles"]) == 1
    assert not update.get("guardrail_events")


async def test_an_empty_allow_list_drops_every_live_article(monkeypatch):
    """12 'Source allow-list': an empty allow-list fails closed: nothing reaches the state."""
    monkeypatch.setattr(graph.settings.tavily, "allowed_domains", [])
    install_news(
        monkeypatch, article("https://www.reuters.com/apple"), article("https://sec.gov/apple")
    )

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    assert update["news_articles"] == []
    assert [event["code"] for event in update["guardrail_events"]] == ["source_dropped"] * 2


async def test_the_node_reads_the_configured_allow_list(monkeypatch):
    """12 'Source allow-list': the list is configuration; the owner's edit takes effect."""
    monkeypatch.setattr(graph.settings.tavily, "allowed_domains", ["example.org"])
    install_news(
        monkeypatch,
        article("https://news.example.org/apple"),
        article("https://www.reuters.com/apple"),
    )

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    assert [item["url"] for item in update["news_articles"]] == ["https://news.example.org/apple"]
    assert update["guardrail_events"] == [{"code": "source_dropped", "detail": "www.reuters.com"}]


async def test_dropped_articles_reach_neither_sentiment_nor_the_citation_registry(monkeypatch):
    """12 'Source allow-list': dropped before sentiment and before the registry, in a real run."""
    install_fakes(monkeypatch)
    install_news(
        monkeypatch,
        article("https://evil.example/a", "OFFLIST-MARKER Apple revenue surged."),
        article("https://www.reuters.com/b", "Apple revenue rose in Europe."),
        article("https://notreuters.com/c", "OFFLIST-MARKER Apple revenue fell."),
    )
    scored: list[str] = []

    def sentiment(texts):
        scored.extend(texts)
        return [SimpleNamespace(label="neutral") for _ in texts]

    monkeypatch.setattr(graph, "analyze_sentiment_batch", sentiment)
    monkeypatch.setattr(graph, "get_llm", lambda _settings: EchoLLM())

    final = await run_agent("AAPL")

    assert scored and any("Apple revenue rose in Europe." in text for text in scored)
    assert not any("OFFLIST-MARKER" in text for text in scored)
    assert "OFFLIST-MARKER" not in json.dumps(final["citation_evidence"])
    assert [item["url"] for item in final["news_articles"]] == ["https://www.reuters.com/b"]
    dropped = [e["detail"] for e in final["guardrail_events"] if e["code"] == "source_dropped"]
    assert dropped == ["evil.example", "notreuters.com"]
    assert FRAMED in final["guardrail_events"]


# --- PII in live news -------------------------------------------------------------------------------

EMAIL = "ir@apple-ir.com"
PHONE = "415-555-0132"
SSN = "123-45-6789"
NEWS_PII = {"code": "pii_redacted", "detail": "news"}


def pii_article(url: str, *, title: str, snippet: str) -> SimpleNamespace:
    return SimpleNamespace(
        title=title, url=url, source="Wire", snippet=snippet, published_date="2026-09-01"
    )


async def test_the_news_node_redacts_pii_in_kept_snippets_and_titles_and_records_each_value(
    monkeypatch,
):
    """12 'Output policy: PII' (live news, owner 2026-11-22): an email, phone or SSN in a kept
    article's snippet or title is replaced with [redacted] at fetch; one pii_redacted event with
    detail `news` per value; no value is left anywhere in the node's update."""
    install_news(
        monkeypatch,
        pii_article(
            "https://www.reuters.com/a",
            title=f"Apple investor line {PHONE}",
            snippet=f"Apple revenue rose; contact {EMAIL} or ref {SSN}.",
        ),
        pii_article(
            "https://www.reuters.com/b",
            title="Apple posts results",
            snippet="Apple revenue rose in Europe.",
        ),
    )

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    first, second = update["news_articles"]
    assert first["title"] == "Apple investor line [redacted]"
    assert first["snippet"] == "Apple revenue rose; contact [redacted] or ref [redacted]."
    assert second["title"] == "Apple posts results"
    assert second["snippet"] == "Apple revenue rose in Europe."
    assert [e for e in update["guardrail_events"] if e["code"] == "pii_redacted"] == [NEWS_PII] * 3
    for value in (EMAIL, PHONE, SSN):
        assert value not in repr(update)


async def test_an_off_list_article_is_dropped_not_redacted(monkeypatch):
    """12 'Output policy: PII' (live news) after 'Source allow-list': redaction applies to kept
    articles; a dropped article records only source_dropped and nothing carries its value."""
    install_news(
        monkeypatch,
        pii_article(
            "https://evil.example/a", title="Apple tip line", snippet=f"Apple revenue; {EMAIL}."
        ),
    )

    update = await graph.research_news_node(create_initial_state("AAPL", "Apple Inc."))

    assert update["news_articles"] == []
    assert [e["code"] for e in update["guardrail_events"]] == ["source_dropped"]
    assert EMAIL not in repr(update)


async def test_news_pii_reaches_neither_sentiment_the_registry_nor_the_prompts(monkeypatch):
    """12 'Output policy: PII' (live news): redacted before sentiment, the citation registry,
    the prompts and the final state, in a real run."""
    install_fakes(monkeypatch)
    install_news(
        monkeypatch,
        pii_article(
            "https://www.reuters.com/a",
            title=f"Apple desk {PHONE}",
            snippet=f"Apple revenue rose, says {EMAIL} (SSN {SSN}).",
        ),
    )
    scored: list[str] = []

    def sentiment(texts):
        scored.extend(texts)
        return [SimpleNamespace(label="neutral") for _ in texts]

    model = CapturingLLM()
    monkeypatch.setattr(graph, "analyze_sentiment_batch", sentiment)
    monkeypatch.setattr(graph, "get_llm", lambda _settings: model)

    final = await run_agent("AAPL")

    assert scored, "premise: the article was scored"
    surfaces = {
        "sentiment input": "\n".join(scored),
        "citation registry": json.dumps(final["citation_evidence"]),
        "prompts": "\n".join(model.prompts),
        "news articles": json.dumps(final["news_articles"]),
        "memo": final["investment_memo"],
    }
    assert model.prompts, "premise: the memo was drafted"
    for name, text in surfaces.items():
        for value in (EMAIL, PHONE, SSN):
            assert value not in text, f"{value} leaked into the {name}"
    news_events = [e for e in final["guardrail_events"] if e["detail"] == "news"]
    assert news_events == [NEWS_PII] * 3


# --- the drafting node ----------------------------------------------------------------------------


async def test_drafting_with_sources_records_that_the_content_was_framed(monkeypatch):
    """12 'Untrusted-content framing': a run with at least one source records
    untrusted_content_framed."""
    monkeypatch.setattr(graph, "get_llm", lambda _settings: CapturingLLM())

    update = await graph.draft_memo_node(evidence_state())

    assert FRAMED in update["guardrail_events"]
    assert all(set(event) == {"code", "detail"} for event in update["guardrail_events"])


async def test_drafting_without_any_source_records_no_framing(monkeypatch):
    """12 'Untrusted-content framing': nothing to frame, nothing recorded."""
    monkeypatch.setattr(graph, "get_llm", lambda _settings: CapturingLLM())

    update = await graph.draft_memo_node(create_initial_state("AAPL", "Apple Inc."))

    assert not update.get("guardrail_events")
