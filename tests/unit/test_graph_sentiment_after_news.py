"""Trace: docs/test-plan.md §10, "Sentiment follows news only (FILINGS-LATENCY, 2026-11-23)": the
sentiment step starts when `research_news` finishes, without waiting for `fetch_stock` or
`retrieve_filings`; with a slow filings fake it is reported finished before the filings step;
`draft_memo` starts only after sentiment, market data and filings have all finished, and the run's
`snapshot_hash` covers all of them; the `freeze_snapshot` join is not a reported progress step, so
`progress` still names only the steps the frontend shows. SPEC §8.1 topology.

The slow branch is a fake that is held until the sentiment step is reported finished (or a bound
runs out), so the order is decided by the graph's wiring and not by a timer. The fakes are the
computing tool fakes of tests/unit/test_graph_fanout.py; the real nodes and the real compiled graph
run. No network, no real LLM.
"""

from __future__ import annotations

import asyncio
import re
import threading
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents import graph
from app.agents.graph import progress_sink, run_agent
from tests.unit.test_graph_fanout import EVIDENCE_STEPS, install_fakes

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPORTED_STEPS = {
    "research_news",
    "fetch_stock",
    "retrieve_filings",
    "analyze_sentiment",
    "draft_memo",
    "verify_memo",
    "revise_memo",
}
HOLD_SECONDS = 2.0  # how long a held branch waits for sentiment before giving up


def _finished(snapshot: list[dict[str, Any]], node: str) -> bool:
    return any(step["node"] == node and step["status"] != "running" for step in snapshot)


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


class HeldStore:
    """The fan-out suite's store, whose first search of either kind blocks (like a vector-store client) until
    `sentiment_done` is set or `HOLD_SECONDS` pass; later searches of the branch run freely."""

    def __init__(self, inner: Any, sentiment_done: threading.Event, released: threading.Event):
        self.inner = inner
        self.sentiment_done = sentiment_done
        self.released = released

    def _hold(self) -> None:
        if not self.released.is_set():
            self.sentiment_done.wait(HOLD_SECONDS)
            self.released.set()

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self._hold()
        return self.inner.search(*args, **kwargs)

    def search_by_ticker(self, *args: Any, **kwargs: Any) -> Any:
        self._hold()
        return self.inner.search_by_ticker(*args, **kwargs)


async def _run_with_branch_held_until_sentiment_finishes(
    monkeypatch: pytest.MonkeyPatch, held: str
) -> tuple[Any, list[list[dict[str, Any]]]]:
    """Run the graph with `held` (`retrieve_filings` or `fetch_stock`) blocked until the progress
    observer has seen `analyze_sentiment` finish, or `HOLD_SECONDS` have passed."""
    install_fakes(monkeypatch)
    loop = asyncio.get_running_loop()
    sentiment_done = threading.Event()
    released = threading.Event()
    sentiment_done_on_loop = asyncio.Event()

    if held == "retrieve_filings":
        build_store = graph.get_vector_store

        def slow_store() -> Any:
            return HeldStore(build_store(), sentiment_done, released)

        monkeypatch.setattr(graph, "get_vector_store", slow_store)
    else:
        quote = graph.get_stock_data

        async def slow_stock(ticker: str) -> Any:
            try:
                await asyncio.wait_for(sentiment_done_on_loop.wait(), HOLD_SECONDS)
            except asyncio.TimeoutError:
                pass
            return await quote(ticker)

        monkeypatch.setattr(graph, "get_stock_data", slow_stock)

    snapshots: list[list[dict[str, Any]]] = []

    def observe(snapshot: list[dict[str, Any]]) -> None:
        snapshots.append(snapshot)
        if _finished(snapshot, "analyze_sentiment"):
            sentiment_done.set()
            loop.call_soon_threadsafe(sentiment_done_on_loop.set)

    token = progress_sink.set(observe)
    try:
        final = await run_agent("AAPL")
    finally:
        progress_sink.reset(token)
    return final, snapshots


@pytest.mark.parametrize("held", ["retrieve_filings", "fetch_stock"])
async def test_sentiment_is_reported_finished_before_a_slow_sibling_branch(
    monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    """test-plan §10: Sentiment follows news only (with a slow filings fake - or a slow market-data
    fake - the sentiment step is reported finished before that step)."""
    final, snapshots = await _run_with_branch_held_until_sentiment_finishes(monkeypatch, held)

    steps = {step["node"]: step for step in snapshots[-1]}
    assert _at(steps["analyze_sentiment"]["ended_at"]) < _at(steps[held]["ended_at"])
    assert _at(steps["analyze_sentiment"]["started_at"]) >= _at(steps["research_news"]["ended_at"])
    assert final["sentiment_result"]["positive_count"] == 10
    assert final["filing_chunks"]
    assert final["stock_data"]["ticker"] == "AAPL"


@pytest.mark.parametrize("held", ["retrieve_filings", "fetch_stock"])
async def test_draft_memo_starts_only_after_sentiment_market_data_and_filings_finished(
    monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    """test-plan §10: Sentiment follows news only (`draft_memo` starts only after sentiment, market
    data and filings have all finished, and none of the three results is missing from the run)."""
    final, snapshots = await _run_with_branch_held_until_sentiment_finishes(monkeypatch, held)

    steps = {step["node"]: step for step in snapshots[-1]}
    drafting_started = _at(steps["draft_memo"]["started_at"])
    for node in ("analyze_sentiment", "fetch_stock", "retrieve_filings"):
        assert _at(steps[node]["ended_at"]) <= drafting_started, node
    assert final["sentiment_result"]
    assert final["stock_data"]
    assert final["filing_chunks"]


async def test_draft_memo_is_not_reported_running_before_the_last_evidence_step_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """test-plan §10: Sentiment follows news only (the join: no snapshot shows `draft_memo` while
    one of sentiment, market data or filings is still running)."""
    _, snapshots = await _run_with_branch_held_until_sentiment_finishes(
        monkeypatch, "retrieve_filings"
    )

    joined = ("analyze_sentiment", "fetch_stock", "retrieve_filings")
    for snapshot in snapshots:
        if any(step["node"] == "draft_memo" for step in snapshot):
            assert all(_finished(snapshot, node) for node in joined)


async def test_progress_names_only_the_reported_steps_and_the_join_is_not_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """test-plan §10: Sentiment follows news only (the `freeze_snapshot` join is not a reported
    progress step, so `progress` still names only the steps the frontend shows)."""
    install_fakes(monkeypatch)
    snapshots: list[list[dict[str, Any]]] = []
    token = progress_sink.set(snapshots.append)
    try:
        await run_agent("AAPL")
    finally:
        progress_sink.reset(token)

    named = {step["node"] for snapshot in snapshots for step in snapshot}
    assert named <= REPORTED_STEPS
    assert named >= EVIDENCE_STEPS | {"analyze_sentiment", "draft_memo", "verify_memo"}
    assert "freeze_snapshot" not in named


def _variant(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    install_fakes(
        monkeypatch,
        failing=frozenset({"fetch_stock"}) if change == "stock" else frozenset(),
        indexed_name=change != "filings",
    )
    if change == "sentiment":
        monkeypatch.setattr(
            graph,
            "analyze_sentiment_batch",
            lambda texts: [SimpleNamespace(label="negative") for _ in texts],
        )


@pytest.mark.parametrize("change", ["sentiment", "stock", "filings"])
async def test_snapshot_hash_covers_sentiment_market_data_and_filings(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """test-plan §10: Sentiment follows news only (the run's `snapshot_hash` covers sentiment,
    market data and filings: changing any one of them changes the hash; the same evidence gives the
    same hash)."""
    install_fakes(monkeypatch)
    baseline = await run_agent("AAPL")
    again = await run_agent("AAPL")
    _variant(monkeypatch, change)

    changed = await run_agent("AAPL")

    assert HEX64.match(baseline["snapshot_hash"])
    assert again["snapshot_hash"] == baseline["snapshot_hash"]
    assert HEX64.match(changed["snapshot_hash"])
    assert changed["snapshot_hash"] != baseline["snapshot_hash"]
