"""

INTEGRATION TESTS FOR WEB SEARCH (LIVE API)

These tests make REAL API calls to Tavily. They verify that:
1. The code works with the actual API (not just mock tests)
2. End-to-end flow works correctly

WARNING: These tests:
- Consume Tavily API quota
- Require network access

Running These Tests:
--------------------
# These are SKIPPED by default
    pytest tests/integration/ -v

# To actually run them:
    pytest tests/integration/ -v --run-integration


"""

import pytest
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.web_search_tool import WebSearchTool, search_news, search_company_news


# Mark ALL tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestTavilyAPIIntegration:
    """Live tests against Tavily API."""
 
    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """
        Ensure API key is available before running tests.
 
        autouse=True means this runs automatically before each test.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key.startswith("tvly-your"):
            pytest.skip("TAVILY_API_KEY not configured - skipping live test")

    def test_api_connection_successful(self):
        """
        Test that we can connect to Tavily API.

        This is a basic health check - if this fails, all other
        integration tests will fail too.
        """
        tool = WebSearchTool()
        result = tool.search("test query", max_results=1)

        # Should not have an error
        assert result.success, f"API call failed: {result.error}"

    def test_search_returns_expected_structure(self):
        """
        Test that Tavily returns data in the expected format.

        This catches API changes that might break our parsing.
        """
        tool = WebSearchTool()
        result = tool.search("Apple stock price", max_results=3)

        assert result.success
        assert len(result.articles) > 0

        # Check first article has expected fields
        article = result.articles[0]
        assert article.title # Non-empty string
        assert article.url.startswith("http")
        assert article.source # Non-empty string
        assert article.snippet # Non-empty string

    def test_search_company_news_returns_relevant_results(self):
        """Test that company-specific search returns relevant results."""
        result = search_company_news("AAPL", "Apple Inc")

        assert result.success
        assert result.has_results

        # At least one result should mention Apple
        all_text = " ".join(
            f"{a.title} {a.snippet}".lower()
            for a in result.articles
        )
        assert "apple" in all_text or "aapl" in all_text

    def test_search_with_max_results_limit(self):
        """Test that max_results parameter is respected."""
        tool = WebSearchTool()

        result = tool.search("technology news", max_results=2)

        assert result.success
        assert len(result.articles) <= 2

    def test_search_execution_time_recorded(self):
        """Test that execution time is properly tracked."""
        tool = WebSearchTool()
        result = tool.search("market update", max_results=1)

        assert result.execution_time_ms > 0
        # Sanity check - should be less than 30 seconds
        assert result.execution_time_ms < 30000


class TestAsyncAPIIntegration:
    """Live async tests against Tavily API."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Ensure API key is available."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key.startswith("tvly-your"):
            pytest.skip("TAVILY_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_async_search_works(self):
        """Test async search method with real API."""
        tool = WebSearchTool()
        result = await tool.search_async("Tesla earnings", max_results=2)

        assert result.success
        assert len(result.articles) > 0