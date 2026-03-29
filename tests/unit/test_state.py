"""Tests for agents.state — state helpers, error tracking, data availability.

Replaces the inline test logic that was in graph.py's old _main() block.
"""

from agents.state import (
    AgentState,
    AgentStep,
    add_error,
    create_initial_state,
    get_context_for_llm,
    get_data_availability,
    has_fatal_error,
)

# Easy 

class TestCreateInitialState:
    def test_ticker_uppercased(self):
        state = create_initial_state("aapl")
        assert state["ticker"] == "AAPL"

    def test_default_fields_initialized(self):
        state = create_initial_state("MSFT")
        assert state["news_articles"] == []
        assert state["stock_data"] == {}
        assert state["errors"] == []
        assert state["current_step"] == AgentStep.RESEARCH_NEWS.value

    def test_company_name_optional(self):
        state = create_initial_state("GOOG")
        assert state["company_name"] == ""

    def test_company_name_provided(self):
        state = create_initial_state("GOOG", company_name="Alphabet Inc.")
        assert state["company_name"] == "Alphabet Inc."


# Medium 

class TestAddError:
    def test_appends_error(self):
        state = create_initial_state("AAPL")
        update = add_error(state, "fetch_stock", "timeout")
        assert len(update["errors"]) == 1
        assert update["errors"][0]["step"] == "fetch_stock"
        assert update["errors"][0]["recoverable"] is True

    def test_non_recoverable_sets_error_step(self):
        state = create_initial_state("AAPL")
        update = add_error(state, "draft_memo", "LLM down", recoverable=False)
        assert update["current_step"] == AgentStep.ERROR.value

    def test_recoverable_preserves_current_step(self):
        state = create_initial_state("AAPL")
        state["current_step"] = AgentStep.FETCH_STOCK.value
        update = add_error(state, "fetch_stock", "rate limited", recoverable=True)
        assert update["current_step"] == AgentStep.FETCH_STOCK.value

    def test_multiple_errors_accumulate(self):
        state = create_initial_state("AAPL")
        update1 = add_error(state, "step1", "err1")
        # Simulate LangGraph merging update1 into state
        state["errors"] = update1["errors"]
        update2 = add_error(state, "step2", "err2")
        assert len(update2["errors"]) == 2


# Hard 

class TestHasFatalError:
    def test_no_errors_returns_false(self):
        state = create_initial_state("AAPL")
        assert has_fatal_error(state) is False

    def test_only_recoverable_returns_false(self):
        state = create_initial_state("AAPL")
        state["errors"] = [{"recoverable": True, "step": "s", "message": "m"}]
        assert has_fatal_error(state) is False

    def test_one_fatal_returns_true(self):
        state = create_initial_state("AAPL")
        state["errors"] = [{"recoverable": False, "step": "s", "message": "m"}]
        assert has_fatal_error(state) is True

    def test_mixed_errors_returns_true(self):
        state = create_initial_state("AAPL")
        state["errors"] = [
            {"recoverable": True, "step": "s1", "message": "m1"},
            {"recoverable": False, "step": "s2", "message": "m2"},
        ]
        assert has_fatal_error(state) is True


class TestDataAvailability:
    def test_all_missing(self):
        state = create_initial_state("AAPL")
        avail = get_data_availability(state)
        assert avail == {
            "has_stock_data": False,
            "has_news": False,
            "has_filings": False,
            "has_sentiment": False,
        }

    def test_all_present(self, sample_agent_state):
        avail = get_data_availability(sample_agent_state)
        assert avail["has_stock_data"] is True
        assert avail["has_news"] is True
        assert avail["has_filings"] is True
        assert avail["has_sentiment"] is True

    def test_partial(self):
        state = create_initial_state("AAPL")
        state["stock_data"] = {"current_price": 100}
        avail = get_data_availability(state)
        assert avail["has_stock_data"] is True
        assert avail["has_news"] is False


# Edge

class TestGetContextForLLM:
    def test_empty_state_returns_empty(self):
        state = create_initial_state("AAPL")
        ctx = get_context_for_llm(state)
        assert ctx == ""

    def test_includes_stock_data(self):
        state = create_initial_state("AAPL")
        state["stock_data"] = {"company_name": "Apple", "current_price": 185.0}
        ctx = get_context_for_llm(state)
        assert "## Stock Data" in ctx
        assert "185" in ctx

    def test_citation_indices_increment(self):
        state = create_initial_state("AAPL")
        state["news_articles"] = [
            {"title": "Article One", "source": "R", "snippet": "First"},
            {"title": "Article Two", "source": "B", "snippet": "Second"},
        ]
        state["filing_chunks"] = [
            {"section": "Risk Factors", "text": "Risk text here"},
        ]
        ctx = get_context_for_llm(state)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx