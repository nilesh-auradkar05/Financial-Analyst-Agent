import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Generator
import sys
from pathlib import Path

# Add Project Root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PYTEST Configuration


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that make real API calls",
    )


def pytest_configure(config):
    """
    Register custom markers

    Markers are labels used to categorize tests.
    This registration prevents pytest from warning about unknown markers.
    """

    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration test (require --run-integration)",
    )
    config.addinivalue_line("markers", "slow: marks tests as slow-running")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on command-line arguments

    This hook executes after all tests are collected but before they are executed
    Used to skip integration tests when --run-integration is not specified

    Args:
        config: pytest config object
        items: list of collected test items
    """
    if config.getoption("--run-integration"):
        return

    # skip tests marked with @pytest.mark.integration
    skip_integration = pytest.mark.skip(
        reason="Integration test - use --run-integration to run"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# Data Fixtures


@pytest.fixture
def sample_tavily_response() -> dict:
    """
    Sample response from Travily API.

    This fixture mimics what Tavily api returns.
    This is just a sample response structure to test the parsing logic and to avoid making real API calls.

    Returns:
        dict: Sample response Dict matching Tavily's reesponse structure.
    """

    return {
        "results": [
            {
                "title": "Apple Reports Record Q4 Earnings",
                "url": "https://www.reuters.com/technology/apple-q4-earnings-2024",
                "content": "Apple Inc reported record quarterly earnings on Thursday, "
                "beating analyst expectations with strong iPhone sales...",
                "score": 0.95,
                "published_date": "2024-01-15",
            },
            {
                "title": "Apple Stock Rises on Earnings Beat",
                "url": "https://finance.yahoo.com/news/apple-stock-earnings",
                "content": "Shares of Apple (AAPL) rose 3% in after-hours trading "
                "following the company's earnings announcement...",
                "score": 0.88,
                "published_date": "2024-01-15",
            },
            {
                "title": "Analysis: What Apple's Earnings Mean for Tech",
                "url": "https://www.bloomberg.com/opinion/apple-earnings-analysis",
                "content": "Apple's strong quarter has implications for the broader "
                "tech sector and investor sentiment...",
                "score": 0.75,
                "published_date": "2024-01-16",
            },
        ]
    }


@pytest.fixture
def sample_empty_response() -> dict:
    """Empty Tavily response from testing edge cases."""
    return {"results": []}


@pytest.fixture
def sample_error_response() -> dict:
    """Malformed response for testing error handling."""
    return {"error": "API rate limit exceeded"}


# MOCK Fixtures


@pytest.fixture
def mock_tavily_response(sample_tavily_response) -> MagicMock:
    """
    A Mock TavilyClient that returns sample data.

    Returns:
        MagicMock configured to behave like TavilyClient
    """

    mock = MagicMock()

    mock.search.return_value = sample_tavily_response

    return mock


@pytest.fixture
def mock_async_tavily_response(sample_tavily_response) -> MagicMock:
    """
    A mock AsyncTavilyClient for async tests.
    """

    mock = MagicMock()
    mock.search = AsyncMock(return_value=sample_tavily_response)

    return mock


# Environment Fixtures


@pytest.fixture
def mock_env_with_api_key(monkeypatch):
    """
    Environment setup with test API key.

    used monkeypatch to temporarily set environment variable.

    Args:
        monkeypatch: pytest's monkeypatch fixture
    """

    monkeypatch.setenv("TAVILY_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-langchain-key")


@pytest.fixture
def mock_env_without_api_key(monkeypatch):
    """Environment setup without API keys for testing error handling."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
