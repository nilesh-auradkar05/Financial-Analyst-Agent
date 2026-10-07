"""Trace: docs/test-plan.md §18, run telemetry (API-RUN-TELEMETRY).

Runs the real nodes and compiled graph over the computing tool fakes from the fan-out suite.
"""

from types import SimpleNamespace

from app.agents import graph
from app.agents.graph import progress_sink, run_agent
from tests.unit.test_graph_fanout import EVIDENCE_STEPS, EchoLLM, install_fakes

NODES = ["analyze_sentiment", "draft_memo", "verify_memo"]


async def observed_run(monkeypatch, **fakes):
    install_fakes(monkeypatch, **fakes)
    snapshots: list[list[dict]] = []
    token = progress_sink.set(snapshots.append)
    try:
        final = await run_agent("AAPL")
    finally:
        progress_sink.reset(token)
    return final, snapshots


async def test_progress_reports_each_node_running_then_finished(monkeypatch):
    _, snapshots = await observed_run(monkeypatch, failing=frozenset({"fetch_stock"}))

    last = {step["node"]: step for step in snapshots[-1]}
    order = [step["node"] for step in snapshots[-1]]
    assert set(order[:3]) == EVIDENCE_STEPS
    assert order[3:] == NODES
    assert last["fetch_stock"]["status"] == "degraded"
    assert last["research_news"]["status"] == last["retrieve_filings"]["status"] == "completed"
    assert all(step["ended_at"] >= step["started_at"] and step["duration_ms"] >= 0 for step in last.values())

    first_seen = {}
    for snapshot in snapshots:
        for step in snapshot:
            first_seen.setdefault(step["node"], step["status"])
    assert set(first_seen.values()) == {"running"}

    first_finish = next(s for s in snapshots if any(step["status"] != "running" for step in s))
    assert {step["node"] for step in first_finish} == EVIDENCE_STEPS


async def test_run_without_observer_returns_the_same_result(monkeypatch):
    observed, _ = await observed_run(monkeypatch)
    plain = await run_agent("AAPL")

    for key in ("investment_memo", "news_articles", "stock_data", "citations", "verification_result"):
        assert plain[key] == observed[key]


async def test_usage_and_summary_come_from_the_memo_response(monkeypatch):
    class ReportingLLM:
        async def ainvoke(self, messages):
            prompt = messages[-1]["content"]
            return SimpleNamespace(
                content="# Memo\n\n## 1. Executive Summary\n\nRevenue grew [1].\n\n## 2. Company Overview\n\nDetails [2].",
                usage_metadata={"input_tokens": len(prompt.split()), "output_tokens": 12},
                response_metadata={"model_name": "echo-1"},
            )

    install_fakes(monkeypatch)
    monkeypatch.setattr(graph, "get_llm", lambda _settings: ReportingLLM())
    reported = await run_agent("AAPL")

    assert reported["executive_summary"] == "Revenue grew [1]."
    assert reported["llm_usage"]["model"] == "echo-1"
    assert reported["llm_usage"]["output_tokens"] == 12
    assert reported["llm_usage"]["input_tokens"] > 100

    monkeypatch.setattr(graph, "get_llm", lambda _settings: EchoLLM())
    silent = await run_agent("AAPL")
    assert silent["llm_usage"] == {"model": None, "input_tokens": None, "output_tokens": None}
