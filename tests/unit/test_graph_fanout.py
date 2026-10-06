"""Trace: docs/test-plan.md §10, concurrency and event-loop hygiene (S2-T00c).

Tool-level fakes compute their outputs from their inputs and run through the real
nodes and the real compiled graph. No network, no FinBERT, no real LLM.
"""

import asyncio
import re
import time
from types import SimpleNamespace

import httpx
import pytest

from app import main as api
from app.agents import graph
from app.agents.graph import run_agent
from app.agents.state import create_initial_state

BRANCH_SECONDS = 1.0
EVIDENCE_STEPS = {"research_news", "fetch_stock", "retrieve_filings"}
FILING_SEARCHES = 5  # four topic queries plus the company-name lookup


class EchoLLM:
    """Writes one cited sentence per numbered source found in the prompt."""

    async def ainvoke(self, messages):
        sources = re.findall(r"\[(\d+)\] [^\n]+\n([^\n]+)", messages[-1]["content"])
        memo = "## Executive Summary\n" + "\n".join(f"{text} [{index}]" for index, text in sources)
        return SimpleNamespace(content=memo)


def branch_errors(final) -> list[str]:
    """Steps of the errors raised by the evidence branches (the verifier adds its own)."""
    return sorted(error["step"] for error in final["errors"] if error["step"] in EVIDENCE_STEPS)


def _label(text: str) -> str:
    return "positive" if "growth" in text else "neutral"


def install_fakes(
    monkeypatch,
    *,
    delay: float = 0.0,
    failing: frozenset[str] = frozenset(),
    indexed_name: bool = True,
) -> None:
    async def news(query, max_results=10):
        await asyncio.sleep(delay)
        if "research_news" in failing:
            raise RuntimeError("news provider down")
        return [
            SimpleNamespace(
                title=f"{query} #{n}",
                url=f"https://example.test/{n}",
                source="Wire",
                snippet=f"{query} item {n} reports revenue growth.",
                published_date="2026-09-01",
            )
            for n in range(max_results)
        ]

    async def stock(ticker):
        await asyncio.sleep(delay)
        if "fetch_stock" in failing:
            raise RuntimeError("quote provider down")
        return SimpleNamespace(
            ticker=ticker,
            company_name=f"{ticker} Corporation",
            current_price=float(sum(map(ord, ticker))),
            price_change_percent=1.0,
            market_cap=1_000_000,
            pe_ratio=10.0,
            fifty_two_week_high=400.0,
            fifty_two_week_low=50.0,
            volume=100,
            dividend_yield=0.0,
            sector="Technology",
            industry="Software",
            recommendation="hold",
        )

    class Store:
        def search_by_ticker(self, query, ticker, n_results=3, section=None):
            time.sleep(delay / FILING_SEARCHES)  # blocking, like a real vector-store client
            metadata = {"filing_type": "10-K"}
            if indexed_name:
                metadata["company_name"] = f"{ticker} Inc."
            return SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        id=f"{ticker}:{query}:{n}",
                        text=f"{n} {query}: filing text for {ticker}.",
                        section="Business",
                        metadata=metadata,
                        filing_date="2026-01-01",
                        relevance_score=1 / (n + 1),
                    )
                    for n in range(n_results)
                ]
            )

    monkeypatch.setattr(graph, "search_company_news", news)
    monkeypatch.setattr(graph, "get_stock_data", stock)
    monkeypatch.setattr(graph, "get_vector_store", Store)
    monkeypatch.setattr(
        graph,
        "analyze_sentiment_batch",
        lambda texts: [SimpleNamespace(label=_label(text)) for text in texts],
    )
    monkeypatch.setattr(graph, "get_llm", lambda _settings: EchoLLM())


async def test_independent_evidence_nodes_fan_out(monkeypatch):
    install_fakes(monkeypatch, delay=BRANCH_SECONDS)

    started = time.perf_counter()
    final = await run_agent("AAPL")
    elapsed = time.perf_counter() - started

    assert len(final["news_articles"]) == 10
    assert final["stock_data"]["company_name"] == "AAPL Corporation"
    assert final["filing_chunks"]
    assert elapsed < 1.6, f"evidence phase took {elapsed:.2f}s; three 1 s branches ran serially"


async def test_one_failed_branch_keeps_the_other_two(monkeypatch):
    install_fakes(monkeypatch, failing=frozenset({"fetch_stock"}))

    final = await run_agent("AAPL")

    assert final["stock_data"] == {}
    assert branch_errors(final) == ["fetch_stock"]
    assert len(final["news_articles"]) == 10
    assert final["filing_chunks"]


async def test_errors_from_two_branches_both_survive(monkeypatch):
    install_fakes(monkeypatch, failing=frozenset({"fetch_stock", "research_news"}))

    final = await run_agent("AAPL")

    assert branch_errors(final) == ["fetch_stock", "research_news"]


async def test_health_answers_while_sentiment_scores_50_snippets(monkeypatch):
    def slow_batch(texts):
        time.sleep(0.6)  # CPU-bound scoring holds its thread
        return [SimpleNamespace(label=_label(text)) for text in texts]

    async def healthy(**_kwargs):
        return True

    async def langsmith_status():
        return {"connected": False}

    monkeypatch.setattr(graph, "analyze_sentiment_batch", slow_batch)
    monkeypatch.setattr(api, "check_ollama_health", healthy)
    monkeypatch.setattr(api, "get_vector_store", lambda: SimpleNamespace(count=0))
    monkeypatch.setattr(api, "check_langsmith_connection", langsmith_status)
    state = create_initial_state("AAPL")
    state["news_articles"] = [{"snippet": f"item {n} reports revenue growth"} for n in range(50)]

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")  # warm-up
        started = time.perf_counter()
        scoring = asyncio.create_task(graph.analyze_sentiment_node(state))
        await client.get("/health")
        elapsed = time.perf_counter() - started
        update = await scoring

    assert update["sentiment_result"]["positive_count"] == 50
    assert elapsed < 0.2, f"/health took {elapsed:.2f}s while sentiment was scoring"


async def test_graph_is_built_once_not_per_request(monkeypatch):
    install_fakes(monkeypatch)
    agent = graph.AGENT

    def rebuild():
        raise AssertionError("create_agent was invoked for a request")

    monkeypatch.setattr(graph, "create_agent", rebuild)
    first = await run_agent("AAPL")
    second = await run_agent("MSFT")

    assert graph.AGENT is agent
    assert first["stock_data"]["ticker"] == "AAPL"
    assert second["stock_data"]["ticker"] == "MSFT"


async def test_every_retrieved_filing_chunk_reaches_the_registry(monkeypatch):
    install_fakes(monkeypatch)

    final = await run_agent("AAPL")

    cited = [entry for entry in final["citation_evidence"] if entry["source_type"] == "sec_filing"]
    assert len(final["filing_chunks"]) == len(cited) > 0


@pytest.mark.parametrize(
    "supplied, indexed_name, expected",
    [
        (None, True, "AAPL Inc."),  # name read from indexed filing metadata
        (None, False, "AAPL"),  # nothing indexed with a name: ticker
        ("Apple Inc.", True, "Apple Inc."),  # caller-supplied name wins
    ],
)
async def test_filing_queries_do_not_depend_on_the_stock_branch(
    monkeypatch, supplied, indexed_name, expected
):
    install_fakes(monkeypatch, failing=frozenset({"fetch_stock"}), indexed_name=indexed_name)

    final = await run_agent("AAPL", supplied)

    assert final["filing_chunks"]
    assert all(f" {expected} " in chunk["text"] for chunk in final["filing_chunks"])
