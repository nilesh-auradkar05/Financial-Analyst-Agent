"""Tests for graceful-degradation routing in agents.graph (item 3).

The ``_route_after_node`` factory generates routers that skip to
draft_memo when a fatal error is present.  These tests verify routing
decisions WITHOUT calling external services.
"""

from agents.graph import _route_after_node
from agents.state import AgentStep, add_error, create_initial_state


class TestRouteAfterNode:
    """Unit-test the router factory."""

    def test_continues_normally_when_no_error(self):
        state = create_initial_state("AAPL")
        router = _route_after_node("fetch_stock")
        assert router(state) == "fetch_stock"

    def test_continues_on_recoverable_error(self):
        state = create_initial_state("AAPL")
        update = add_error(state, "research_news", "timeout", recoverable=True)
        state["errors"] = update["errors"]
        router = _route_after_node("fetch_stock")
        assert router(state) == "fetch_stock"

    def test_skips_to_memo_on_fatal_error(self):
        state = create_initial_state("AAPL")
        update = add_error(state, "research_news", "critical failure", recoverable=False)
        state["errors"] = update["errors"]
        router = _route_after_node("fetch_stock")
        assert router(state) == "draft_memo"

    def test_skips_regardless_of_target(self):
        """Every edge should skip to draft_memo on fatal, not to its own target."""
        state = create_initial_state("AAPL")
        state["errors"] = [{"recoverable": False, "step": "x", "message": "y"}]

        for target in ["fetch_stock", "retrieve_filings", "analyze_sentiment"]:
            router = _route_after_node(target)
            assert router(state) == "draft_memo"

    def test_multiple_recoverable_still_continues(self):
        state = create_initial_state("AAPL")
        state["errors"] = [
            {"recoverable": True, "step": "a", "message": "m"},
            {"recoverable": True, "step": "b", "message": "m"},
        ]
        router = _route_after_node("retrieve_filings")
        assert router(state) == "retrieve_filings"

    def test_mixed_errors_skips(self):
        """One fatal among many recoverable → skip."""
        state = create_initial_state("AAPL")
        state["errors"] = [
            {"recoverable": True, "step": "a", "message": "m"},
            {"recoverable": False, "step": "b", "message": "m"},
            {"recoverable": True, "step": "c", "message": "m"},
        ]
        router = _route_after_node("analyze_sentiment")
        assert router(state) == "draft_memo"
