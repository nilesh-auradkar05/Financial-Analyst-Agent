"""
ALPHA ANALYST - EVALUATION TESTS
================================
Minimal tests for evaluation heuristics.

Run: pytest tests/test_eval.py -v
"""

import pytest

from evaluation import EvalResult, check_memo

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def good_memo():
    return """
    # Executive Summary

    Apple Inc. (AAPL) demonstrates strong fundamentals.

    ## Company Overview

    Apple designs iPhones, Macs, iPads, and services.
    Revenue: $394 billion. Gross margin: 44%.

    ## Risk Factors

    Supply chain concentration in China. Regulatory scrutiny.

    ## Financial Highlights

    - Revenue growth: 8%
    - Net income: $97B
    - Free cash flow: $99B

    ## Conclusion

    Rating: BUY with target of $210. [1][2]
    """


@pytest.fixture
def bad_memo():
    return "Buy the stock."


# =============================================================================
# TESTS
# =============================================================================


class TestCheckMemo:
    """Tests for check_memo heuristic."""

    def test_good_memo_passes(self, good_memo):
        score, issues = check_memo(good_memo, "AAPL")
        assert score >= 0.7
        assert len(issues) <= 1

    def test_bad_memo_fails(self, bad_memo):
        score, issues = check_memo(bad_memo, "AAPL")
        assert score < 0.5
        assert "short" in str(issues).lower()

    def test_missing_ticker_flagged(self, good_memo):
        memo = good_memo.replace("AAPL", "the company").replace("Apple", "Company")
        score, issues = check_memo(memo, "AAPL")
        assert "ticker" in str(issues).lower()

    def test_no_citations_flagged(self):
        memo = "AAPL is good. " * 50  # Long enough but no citations
        score, issues = check_memo(memo, "AAPL")
        assert "citation" in str(issues).lower()


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_to_dict(self):
        r = EvalResult(ticker="AAPL", passed=True, score=0.85)
        d = r.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["passed"] is True
        assert d["score"] == 0.85

    def test_defaults(self):
        r = EvalResult(ticker="TEST")
        assert r.passed is False
        assert r.score == 0.0
        assert r.error is None


# =============================================================================
# INTEGRATION (skip without agent module)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_evaluate_ticker():
    """Integration test - requires agent module."""
    try:
        from evaluation import evaluate_ticker
        result = await evaluate_ticker("AAPL", run_full=False)
        assert result.ticker == "AAPL"
        # May or may not pass depending on environment
    except ImportError:
        pytest.skip("Agent module not available")
