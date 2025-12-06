"""
WEB SEARCH TOOL (TAVILY)

This module provides web search capabilities using Tavily API.

Usage:
    from tools.web_search_tool import search_news, NewsArticle

    articles = await search_news("Apple earning Q2 2025")
    for article in articles:
        print(f"{article.title} ({article.source})")
        print(f"    URL: {article.url}")
        print(f"    Published: {article.published_date}")
"""

from multiprocessing.sharedctypes import Value
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from loguru import logger

from tavily import TavilyClient, AsyncTavilyClient

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings


@dataclass
class NewsArticle:
    """
    Represents a single news article with full provenance for citations

    Citation Requirements:
        - url: To click and read the original source
        - source: To assess credibility
        - published_date: To confirm recency/relevance
        - title: To identify the article

    Analysis Requirements:
    For agent to Analyze the article:
        - snippet: The actual text content to process
        - relevance_score: To prioritize which articles matter most

    Attributes:
        title: Article headline
        url: Direct link to the article (REQUIRED FOR CITATION)
        source: Publisher name (eg: "Bloomberg")
        published_date: When the article was published
        snippet: full content from the article
        relevance_score: Tavily's relevance score (0.0 to 1.0)
        author: Article author if available
        raw_content: Full article content if fetched (for deep analysis)
    """

    title: str
    url: str
    source: str
    snippet: str

    published_date: Optional[str] = None
    relevance_score: float = 0.0
    author: Optional[str] = None
    raw_content: Optional[str] = None

    fetched_at: str = field(
        default_factory=lambda: datetime.now().isoformat(),
    )

    def to_citation(self, index: int) -> str:
        """
        Format this articles as a citation string.

        Args:
            index: Citation number

        Returns:
            Formated citation string

        Example:
            "[1] Apple Reports Strong Q3 - Bloomberg, 2024-01-15
                 https://www.bloomberg.com/..."
        """
        date_str = f", {self.published_date}" if self.published_date else ""
        return f"[{index}] {self.title} - {self.source}{date_str}\n      {self.url}"

    def to_context(self) -> str:
        """
        Format this article as context for the LLM.

        Returns a structured string that helps the LLM understand
        the source while analyzing content.

        Returns:
            Formatted context string
        """
        return f"""Source: {self.source}
                   Title: {self.title}
                   Date: {self.published_date or "Unknown"}
                   URL: {self.url}
                   
                   Content:
                   {self.snippet}"""


@dataclass
class SearchResult:
    """
    Container for a complete search operation result.

    This wraps multiple NewsArticle along with metadata about
    the search itself.

    Attributes:
        query: The search query that was exceuted
        articles: List of NewsArticle objects
        total_results: Total number of results found
        search_depth: "basic" or "advanced"
        execution_time_ms: Total time taken for search
        error: Error message if search failed
    """

    query: str
    articles: list[NewsArticle]
    total_results: int = 0
    search_depth: str = "advanced"
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return if search was successful"""
        return self.error is None

    @property
    def has_results(self) -> bool:
        """Return if search has results"""
        return len(self.articles) > 0


# Search Client


class WebSearchTool:
    """
    Web search tool using Tabily API.

    This class provides a clean interface for searching the web
    and returns structured results suitable for agent analysis.

    Features:
        - Automatic retry on transient failures
        - Structured logging for debugging
        - Converts raw API responses to NewsArticle objects
        - Supports both sync and async interfaces

    Example:
        # Initialize
        search_tool = WebSearchTool()

        # Synchronous search
        result = search_tool.search("Apple stock news")

        # Asynchronous search
        results = await search_tool.search("Apple stock news")

        # check results
        if result.success:
            for article in result.articles:
                print(article.title)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Inititalize the web search tool.

        Args:
            api_key: Tavily API key. If not provided, uses settings.tavily_api_key (from env file).

        Raises:
            ValueError: If no API key is available.
        """

        self.api_key = api_key or settings.tavily.api_key

        if not self.api_key:
            logger.warning(
                "No Tavily API key provided. Web search will not work."
                "Set TAVILY_API_KEY in your .env file."
            )
        self._client = None
        self._async_client = None
       

        self.max_results = settings.tavily.max_results
        self.search_depth = settings.tavily.search_depth

        logger.info(
            "WebSearchTool initialized"
            f"(max_results={self.max_results}, depth={self.search_depth})"
        )

    @property
    def client(self) -> TavilyClient:
        """Lazy initalization of sync client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Tavily API key not configured.")
            self._client = TavilyClient(api_key=self.api_key)

        return self._client

    @property
    def async_client(self) -> AsyncTavilyClient:
        """Lazy initialization of async client."""
        if self._async_client is None:
            if not self.api_key:
                raise ValueError("Tavily API key not configured.")

            self._async_client = AsyncTavilyClient(api_key=self.api_key)

        return self._async_client

    def _parse_result(self, raw_result: dict) -> NewsArticle:
        """
        Convert a raw Tavily result to a NewsArticle.

        Tavily return results in below format:
        {
            "title": "....",
            "url": "....",
            "content": "....",
            "score": number,
            "published_date: date",
            "raw_content": "....",
        }

        Args:
            raw_result: Single result dict from Tavily API

        Returns:
            NewsArticle object with cleaned/normalized data
        """

        # Extract domain from URL as source name
        # eg: https://www.bloomberg.com/... -> "Bloomberg"
        url = raw_result.get("url", "")
        source = self._extract_source(url)

        return NewsArticle(
            title=raw_result.get("title", "Untitled"),
            url=url,
            source=source,
            snippet=raw_result.get("content", ""),
            published_date=raw_result.get("published_date"),
            relevance_score=raw_result.get("score", 0),
            raw_content=raw_result.get("raw_content"),
        )

    def _extract_source(self, url: str) -> str:
        """
        Extract a clean source name from a URL.

        Examples:
            "https://www.bloomberg.com/..." -> "Bloomberg"
            "https://www.forbes.com/..." -> "Forbes"
            "https://www.wsj.com/..." -> "Wall Street Journal"

        Args:
            url: Full URL string

        Returns:
            Cleaned source name
        """

        known_sources = {
            "reuters.com": "Reuters",
            "forbes.com": "Forbes",
            "wsj.com": "Wall Street Journal",
            "bloomberg.com": "Bloomberg",
            "cnbc.com": "CNBC",
            "marketwatch.com": "MarketWatch",
            "seekingalpha.com": "Seeking Alpha",
            "finviz.com": "Finviz",
            "investing.com": "Investing.com",
            "tradingview.com": "TradingView",
            "finance.yahoo.com": "Yahoo Finance",
            "yahoo.com": "Yahoo",
            "fool.com": "Motley Fool",
            "ft.com": "Financial Times",
            "investopedia.com": "Investopedia",
            "sec.gov": "SEC",
            "nytimes.com": "The New York Times",
            "washingtongpost.com": "The Washington Post",
            "bbc.com": "BBC",
            "cnn.com": "CNN",
            "techcrunch.com": "TechCrunch",
            "theverge.com": "The Verge",
            "wired.com": "Wired",
        }

        try:
            # Extract domain from URL
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Removing www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # check known sources first
            for key, name in known_sources.items():
                if key in domain:
                    return name

            # Fallback to domain name
            return domain

        except Exception:
            return "Unknown Source"

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_raw_content: bool = False,
    ) -> SearchResult:
        """
        Perform a Synchronous web search using Tavily API.

        Args:
            query: Search query string
            max_results: Override default max results
            search_depth: Override default search depth
            include_raw_content: Whether to fetch full article content

        Returns:
            SearchResult containing list of NewsArticle objects

        Example:
            result = search_tool.search("Tesla stock news")
            if result.success:
                for article in result.articles:
                    print(f"{article.title}  ({article.source})")
        """

        import time

        start_time = time.time()

        # Use provided values or fall back to default settings
        max_results = max_results or self.max_results
        search_depth = search_depth or self.search_depth

        logger.info(
            f"Searching: '{query}' (max_results={max_results}, depth={search_depth})"
        )

        try:
            # Make API call
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )

            # Parse results
            articles = [self._parse_result(r) for r in response.get("results", [])]

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Search completed: {len(articles)} results in {execution_time:.1f} ms"
            )

            return SearchResult(
                query=query,
                articles=articles,
                total_results=len(articles),
                search_depth=search_depth,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Search failed: {e}")

            return SearchResult(
                query=query,
                articles=[],
                execution_time_ms=execution_time,
                error=str(e),
            )

    async def search_async(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_raw_content: bool = False,
    ) -> SearchResult:
        """
        Perform an asynchronous web search using Tavily API.

        Args:
            query: Search query string
            max_results: Override default max results
            search_depth: Override default search depth
            include_raw_content: Whether to fetch full article content

        Returns:
            SearchResult containing list of NewsArticle objects
        """

        import time

        start_time = time.time()

        max_results = max_results or self.max_results
        search_depth = search_depth or self.search_depth

        logger.info(f"Async searching: '{query}'")

        try:
            response = await self.async_client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )

            articles = [self._parse_result(r) for r in response.get("results", [])]

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Async search completed: {len(articles)} results in {execution_time:.1f} ms"
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Async search failed: {e}")

            return SearchResult(
                query=query,
                articles=[],
                execution_time_ms=execution_time,
                error=str(e),
            )


# Singleton function for convenience
_default_tool: Optional[WebSearchTool] = None


def _get_default_tool() -> WebSearchTool:
    """Get or create the default WebSearchtool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = WebSearchTool()
    return _default_tool


def search_news(query: str, max_results: int = 5) -> SearchResult:
    """
    Search for news Articles (Synchronous search wrapper)

    Args:
        query: Search query string
        max_results: Maxium number of results to return

    Returns:
        SearchResult with list of NewsArticle Objects

    Example:
        from tools.web_search_tool import search_news

        result = search_news("Tesla Q4 earnings)
        for article in result.articles:
            print(article.to_citation(1))
    """

    tool = _get_default_tool()
    return tool.search(query, max_results=max_results)


async def search_news_async(query: str, max_results: int = 5) -> SearchResult:
    """
    Search for news Articles (Asynchronous search wrapper)

    Args:
        query: Search query string
        max_results: Maxium number of results to return

    Returns:
        SearchResult with list of NewsArticle Objects

    Example:
        from tools.web_search_tool import search_news_async

        result = await search_news_async("Tesla Q4 earnings)
        for article in result.articles:
            print(article.to_citation(1))
    """

    tool = _get_default_tool()
    return await tool.search_async(query, max_results=max_results)


def search_company_news(
    ticker: str,
    company_name: str,
    days_back: int = 7,
) -> SearchResult:
    """
    Search for recent news about a specific company.

    This contructs an optimized query for financial news search.

    Args:
        ticker: Stock ticker symbol (eg: "AAPL")
        company_name: Full company name (eg: "Apple Inc.")
        days_back: How many days of news to look for (default: 7)

    Returns:
        SearchResult with relevant news articles

    Example:
        result = search_company_news("AAPL", "Apple Inc.")
        print(f"Found {len(result.articles)} articles")
    """

    # Construct a query that targets financial news
    # Including both ticker and company name imporves relevance
    query = f"{company_name} ({ticker}) stock news financial"

    logger.info(f"Searching companu news for {ticker}: '{query}'")

    tool = _get_default_tool()
    return tool.search(query, search_depth="advanced")


""" TESTING CLI """
if __name__ == "__main__":
    """
    Test the web search tool from command line.

    Usage:
        - python -m tool.web_search_tool "Apple earnings"

    Or Simply:
        - python tools/web_search_tool.py
    """

    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Apple stock news"

    print(f"\n{'=' * 60}")
    print("Testomg Web Search Tool")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    result = search_news(query)

    if result.success:
        print(
            f"Found {len(result.articles)} articles in {result.execution_time_ms:.1f} ms"
        )

        for i, article in enumerate(result.articles, 1):
            print(f"{'-' * 60}")
            print(article.to_citation(1))
            print(f"Relevance: {article.relevance_score:.2f}")
            print(f"\nSnippet: {article.snippet[:200]}...")
            print()

    else:
        print(f"Search failed: {result.error}")
