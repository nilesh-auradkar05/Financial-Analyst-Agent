"""
Shared test configuration and fixtures.

No sys.path hacks — relies on ``pip install -e .`` (item 12).
"""

import pytest

# ── CLI options & markers ───────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--run-integration", action="store_true", default=False, help="Run integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (requires --run-slow)")
    config.addinivalue_line("markers", "integration: marks tests requiring external services")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="Need --run-slow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Need --run-integration option")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


# ── Shared fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_ticker():
    return "AAPL"


@pytest.fixture
def sample_stock_data():
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 185.50,
        "previous_close": 184.25,
        "market_cap": 2900000000000,
        "pe_ratio": 28.5,
        "dividend_yield": 0.005,
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 164.08,
        "volume": 54000000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }


@pytest.fixture
def sample_news_articles():
    return [
        {
            "title": "Apple Reports Record Q4 Earnings",
            "snippet": "Apple Inc. reported record quarterly revenue of $94.8B.",
            "url": "https://example.com/apple-q4",
            "source": "Reuters",
            "published_date": "2024-01-15",
        },
        {
            "title": "iPhone 16 Sales Exceed Expectations",
            "snippet": "The latest iPhone models are selling faster than prior launches.",
            "url": "https://example.com/iphone-sales",
            "source": "Bloomberg",
            "published_date": "2024-01-14",
        },
        {
            "title": "Apple Services Revenue Hits New High",
            "snippet": "Apple's services segment reached $23B in quarterly revenue.",
            "url": "https://example.com/services",
            "source": "CNBC",
            "published_date": "2024-01-13",
        },
    ]


@pytest.fixture
def sample_filing_chunks():
    return [
        {
            "text": "Apple Inc. designs, manufactures, and markets smartphones.",
            "section": "Business",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
            "relevance_score": 0.92,
        },
        {
            "text": "Global economic conditions could adversely affect the Company.",
            "section": "Risk Factors",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
            "relevance_score": 0.88,
        },
    ]


@pytest.fixture
def sample_sentiment_result():
    return {
        "overall_sentiment": "positive",
        "positive_count": 2,
        "negative_count": 0,
        "neutral_count": 1,
    }


@pytest.fixture
def sample_agent_state(
    sample_ticker, sample_stock_data, sample_news_articles,
    sample_filing_chunks, sample_sentiment_result,
):
    """A fully-populated agent state for testing downstream consumers."""
    return {
        "ticker": sample_ticker,
        "company_name": "Apple Inc.",
        "stock_data": sample_stock_data,
        "news_articles": sample_news_articles,
        "filing_chunks": sample_filing_chunks,
        "sentiment_result": sample_sentiment_result,
        "investment_memo": "# Memo\n\n## Executive Summary\nStrong [1] performance [2].",
        "executive_summary": "Strong performance across segments.",
        "citations": [
            {"index": 1, "type": "news", "title": "Apple Q4 Earnings", "url": "https://example.com"},
            {"index": 2, "type": "sec_filing", "title": "Risk Factors — 10-K"},
        ],
        "errors": [],
        "current_step": "complete",
        "started_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:01:30Z",
        "execution_time_ms": 90000.0,
    }


@pytest.fixture
def mock_ollama_response():
    class MockResponse:
        content = "This is a mock LLM response for testing."
    return MockResponse()


@pytest.fixture
def mock_embedding_vector():
    import random
    random.seed(42)
    return [random.random() for _ in range(2560)]