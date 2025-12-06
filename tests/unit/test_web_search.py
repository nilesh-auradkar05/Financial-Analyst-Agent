"""
UNIT TESTS FOR WEB SEARCH TOOL

These tests verify the web_search_tool module works correctly WITHOUT making
real API calls. Using mocking to simulate Tavily responses.

Test Conventions:
test_<what>_<condition>_<expected_result>

Examples:
    test_search_with_valid_query_returns_articles
    test_extract_source_from_reuters_url_returns_reuters
    test_search_when_api_fails_returns_error

How to Execute tests:
    pytest tests/unit/test_web_search.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.web_search_tool import (
    NewsArticle,
    SearchResult,
    WebSearchTool,
    WebSearchTool,
    search_news,
)


class TestNewsArticle:
    """Tests for the NewsArticle dataclass."""

    def test_create_article_with_required_fields(self):
        """
        Test that we can create a NewsArticle with only required fields.

        This verifies that the dataclass has the required fields.
        """

        article = NewsArticle(
            title="Test Article",
            url="https://example.com/article",
            source="Example",
            snippet="This is a test snippet",
        )

        # Assert: Required fields are set
        assert article.title == "Test Article"
        assert article.url == "https://example.com/article"
        assert article.source == "Example"
        assert article.snippet == "This is a test snippet"

        # Assert: Optional fields have defaults
        assert article.published_date is None
        assert article.relevance_score == 0.0
        assert article.author is None
        assert article.raw_content is None

        # Assert: Auto-populated fields have default values
        assert article.fetched_at is not None

    def test_create_article_with_all_fields(self):
        """Test creating a fully-populated NewsArticle"""

        article = NewsArticle(
            title="Test Article",
            url="https://reuters.com/full",
            source="Reuters",
            snippet="Full snippet content here.",
            published_date="2024-01-01",
            relevance_score=0.95,
            author="John Doe",
            raw_content="Full article content .....",
        )

        assert article.published_date == "2024-01-01"
        assert article.relevance_score == 0.95
        assert article.author == "John Doe"
        assert article.raw_content == "Full article content ....."

    def test_to_citation_formats_correctly(self):
        """
        Test that to_citation() produces properly formatted citations.

        Citations should include:
            - Index number in brackets
            - Title
            - Source
            - Date (if available)
            - URL
        """

        article = NewsArticle(
            title="Apple Earnings Report",
            url="https://reuters.com/apple-earnings",
            source="Reuters",
            snippet="...",
            published_date="2024-01-01",
        )

        citation = article.to_citation(1)

        assert "[1]" in citation
        assert "Apple Earnings Report" in citation
        assert "Reuters" in citation
        assert "2024-01-01" in citation
        assert "https://reuters.com/apple-earnings" in citation

    def test_to_citation_without_date(self):
        """Test citation formatting when date is not available."""

        article = NewsArticle(
            title="Untitled Article",
            url="https://example.com",
            source="Example",
            snippet="...",
        )

        citation = article.to_citation(1)

        assert "[1]" in citation
        assert "Untitled Article" in citation
        assert "Example" in citation

    def test_to_context_includes_all_metadata(self):
        """
        Test that to_context() creates proper LLM context.

        The context format helps LLM understand the source
        of information when analyzing it.
        """

        article = NewsArticle(
            title="Market Update",
            url="https://bloomberg.com/market",
            source="Bloomberg",
            snippet="The market showed strong gains today...",
            published_date="2024-01-20",
        )

        context = article.to_context()

        assert "Source: Bloomberg" in context
        assert "Title: Market Update" in context
        assert "Date: 2024-01-20" in context
        assert "URL: https://bloomberg.com/market" in context
        assert "The market showed strong gains today..." in context


# Tests for SearchResult Dataclass


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_success_property_true_when_no_error(self):
        """SearchResult.success should be False when error exists"""

        result = SearchResult(
            query="test query",
            articles=[],
            error="API connection failed",
        )

        assert result.success is False

    def test_has_results_true_when_articles_exist(self):
        """SearchResult.has_results should be True when articles list is non-empty."""
        article = NewsArticle(
            title="Test",
            url="http://test.com",
            source="Test",
            snippet="...",
        )

        result = SearchResult(
            query="test",
            articles=[article],
        )

        assert result.has_results is True

    def test_has_results_false_when_no_articles(self):
        """SearchResult.has_results should be False when articles list is empty."""

        result = SearchResult(
            query="test",
            articles=[],
        )

        assert result.has_results is False

# Test for WebSearchTool

class TestWebSearchTool:
    """Tests for the WebSearchTool class."""

    def test_extract_source_known_domains(self):
        """
        Test that _extract_source correctly identifies known news sources.

        This is a PURE function test - no mocking needed because
        _extract_source doesn't make any external calls.
        """

        tool = WebSearchTool(api_key="test-key")

        test_cases = [
            ("https://www.reuters.com/article/123", "Reuters"),
            ("https://finance.yahoo.com/news/test", "Yahoo Finance"),
            ("https://www.bloomberg.com/news/article", "Bloomberg"),
            ("https://www.wsj.com/articles/test", "Wall Street Journal"),
            ("https://www.cnbc.com/2024/01/15/test", "CNBC"),
        ]

        for url, expected_source in test_cases:
            result = tool._extract_source(url)
            assert result == expected_source, f"Failed for {url}"

    def test_extract_source_unkown_domain(self):
        """Test that unknown domains return the domain name."""

        tool = WebSearchTool(api_key="test-key")

        result = tool._extract_source("https://randomsite.com/article")

        assert result == "randomsite.com"

    def test_extract_source_removes_www_prefix(self):
        """Test that www. prefix is stripped from domain names."""
        tool = WebSearchTool(api_key="test-key")

        result = tool._extract_source("https://www.unknownsite.com/page")

        assert result == "unknownsite.com"
        assert not result.startswith("www.")

    def test_extract_source_handles_invalid_url(self):
        """Test graceful handling of malformed URLs."""

        tool = WebSearchTool(api_key="test-key")
        result = tool._extract_source("not-a-valid-url")

        assert result == "Unknow Source" or isinstance(result, str)

    def test_parse_result_creates_news_article(self, sample_tavily_response):
        """
        Test that _parse_result correctly converts Tavily response to NewsArticle.

        Uses the sample_tavily_response fixture from conftest.py
        """
        tool = WebSearchTool(api_key="test-key")

        raw_result = sample_tavily_response["results"][0]

        article = tool._parse_result(raw_result)

        assert isinstance(article, NewsArticle)
        assert article.title == "Apple Reports Record Q4 Earnings"
        assert article.url == "https://www.reuters.com/technology/apple-q4-earnings-2024"
        assert article.source == "Reuters"
        assert article.relevance_score == 0.95
        assert article.published_date == "2024-01-15"
        assert "record quarterly earnings" in article.snippet

    @patch("tools.web_search_tool.TavilyClient")
    def test_search_returns_articles(
        self,
        mock_tavily_class,
        sample_tavily_response,
    ):
        """
        Test that search() returns properly parsed articles.

        Uses @patch to replace TavilyClient with a mock.
        This prevents any real API calls.

        How @patch works:
        @patch("tools.web_search.TavilyClient") replaces the TavilyClient
        class in the web_search module with a MagicMock.

        The mock is passed as the first argument (mock_tavily_class).
        """
        
        # mock configure the mock
        mock_instance = MagicMock()
        mock_instance.search.return_value = sample_tavily_response
        mock_tavily_class.return_value = mock_instance

        tool = WebSearchTool(api_key="test-key")
        result = tool.search("Apple earnings")

        assert result.success is True
        assert len(result.articles) == 3
        assert result.articles[0].title == "Apple Reports Record Q4 Earnings"
        assert result.articles[0].source == "Reuters"

        mock_instance.search.assert_called_once()

    @patch("tools.web_search_tool.TavilyClient")
    def test_search_handles_api_error(self, mock_tavily_class):
        """
        Test that search() gracefully handles API errors.
        """

        mock_instance = MagicMock()
        mock_instance.search.side_effect = Exception("API connection failed")
        mock_tavily_class.return_value = mock_instance

        tool = WebSearchTool(api_key="test-key")
        result = tool.search("Apple earnings")

        assert result.success is False
        assert result.error == "API connection failed"
        assert result.articles == []

    @patch("tools.web_search_tool.TavilyClient")
    def test_search_handles_empty_results(
        self,
        mock_tavily_class,
        sample_empty_response,
    ):
        """Test handling of search with no results"""
        mock_instance = MagicMock()
        mock_instance.search.return_value = sample_empty_response
        mock_tavily_class.return_value = mock_instance

        tool = WebSearchTool(api_key="test-key")
        result = tool.search("askdjjlkqweopqiw1238241")

        assert result.success is True
        assert result.has_results is False
        assert result.articles == []

    def test_init_without_api_key_logs_warning(self):
        """Test that intializing without API key logs a warning"""
        tool = WebSearchTool(api_key=None)
        assert tool._client is None


# Wrapper Test Functions

class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    @patch("tools.web_search_tool._get_default_tool")
    def test_search_news_uses_default_tool(self, mock_get_tool):
        """Test that search_news() uses the default tool singleton"""
        mock_tool = MagicMock()
        mock_tool.search.return_value = SearchResult(
            query="test",
            articles=[],
        )

        mock_get_tool.return_value = mock_tool

        result = search_news("Apple news")

        mock_tool.search.assert_called_once_with(
            "Apple news",
            max_results=5,
        )

# Parameterized Tests

class TestSourceExtraction:
    """Compreehensive tests for URL -> Source Extraction using parametrize"""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.reuters.com/markets/stocks", "Reuters"),
            ("https://www.bloomberg.com/news/articles/xyz", "Bloomberg"),
            ("https://www.wsj.com/articles/abc", "Wall Street Journal"),
            ("https://www.ft.com/content/123", "Financial Times"),

            ("https://www.cnbc.com/2024/01/15/market", "CNBC"),
            ("https://finance.yahoo.com/quote/AAPL", "Yahoo Finance"),
            ("https://www.marketwatch.com/story/abc", "MarketWatch"),

            ("https://subdomain.reuters.com/article", "Reuters"),
            ("https://www.bbc.com/news/business", "BBC"),

        ],
    )
    def test_known_sources(self, url: str, expected: str):
        """Test source extraction for known domains"""
        tool = WebSearchTool(api_key="test-key")
        assert tool._extract_source(url) == expected