"""
Financial Analyst Agent System Test Configuration
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CLI OPTIONS
# =============================================================================

def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that execute the agent",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests requiring external services",
    )


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (requires --run-slow)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Skip marked tests unless explicitly enabled."""
    # Skip slow tests
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip integration tests
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration option")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def sample_ticker():
    """Default test ticker."""
    return "AAPL"


@pytest.fixture
def sample_stock_data():
    """Sample stock data fixture."""
    return {
        "ticker": "AAPL",
        "current_price": 185.50,
        "previous_close": 184.25,
        "market_cap": 2900000000000,
        "pe_ratio": 28.5,
        "dividend_yield": 0.005,
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 164.08,
        "volume": 54000000,
        "average_volume": 58000000,
    }


@pytest.fixture
def sample_news_articles():
    """Sample news articles fixture."""
    return [
        {
            "title": "Apple Reports Record Q4 Earnings",
            "snippet": "Apple Inc. reported record quarterly revenue...",
            "url": "https://example.com/apple-q4",
            "source": "Reuters",
            "published_date": "2024-01-15",
        },
        {
            "title": "iPhone 16 Sales Exceed Expectations",
            "snippet": "The latest iPhone models are selling faster...",
            "url": "https://example.com/iphone-sales",
            "source": "Bloomberg",
            "published_date": "2024-01-14",
        },
        {
            "title": "Apple Services Revenue Hits New High",
            "snippet": "Apple's services segment continues to grow...",
            "url": "https://example.com/services",
            "source": "CNBC",
            "published_date": "2024-01-13",
        },
    ]


@pytest.fixture
def sample_filing_chunks():
    """Sample SEC filing chunks fixture."""
    return [
        {
            "text": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.",
            "section": "Business",
            "ticker": "AAPL",
            "filing_type": "10-K",
        },
        {
            "text": "The Company's products include iPhone, Mac, iPad, and Wearables, Home and Accessories.",
            "section": "Business",
            "ticker": "AAPL",
            "filing_type": "10-K",
        },
        {
            "text": "Global and regional economic conditions could materially adversely affect the Company.",
            "section": "Risk Factors",
            "ticker": "AAPL",
            "filing_type": "10-K",
        },
    ]


@pytest.fixture
def sample_sentiment_result():
    """Sample sentiment analysis result fixture."""
    return {
        "label": "positive",
        "confidence": 0.85,
        "scores": {
            "positive": 0.85,
            "negative": 0.05,
            "neutral": 0.10,
        },
    }


@pytest.fixture
def sample_agent_state(
    sample_ticker,
    sample_stock_data,
    sample_news_articles,
    sample_filing_chunks,
    sample_sentiment_result,
):
    """Complete sample agent state fixture."""
    return {
        "ticker": sample_ticker,
        "stock_data": sample_stock_data,
        "news_articles": sample_news_articles,
        "filing_chunks": sample_filing_chunks,
        "sentiment_result": sample_sentiment_result,
        "executive_summary": "Apple demonstrates strong fundamentals...",
        "investment_memo": """
# Investment Memo: AAPL

## Executive Summary
Apple Inc. (AAPL) continues to demonstrate market leadership...

## Company Overview
Apple designs and sells consumer electronics...

## Financial Analysis
Revenue of $394B with 44% gross margin...

## Risk Factors
Supply chain and regulatory risks...

## Conclusion
Rating: BUY [1][2][3]
        """,
        "citations": [
            {"id": 1, "source": "SEC 10-K", "text": "Official filing"},
            {"id": 2, "source": "Reuters", "text": "News article"},
            {"id": 3, "source": "YFinance", "text": "Stock data"},
        ],
        "errors": [],
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama LLM response."""
    class MockResponse:
        content = "This is a mock LLM response for testing."
    return MockResponse()


@pytest.fixture
def mock_embedding_vector():
    """Mock embedding vector (4096 dimensions for qwen3-embedding:4b)."""
    import random
    random.seed(42)
    return [random.random() for _ in range(4096)]
