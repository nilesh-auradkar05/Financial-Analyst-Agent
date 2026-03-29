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

from dataclasses import dataclass
from typing import Optional

from langsmith import traceable
from loguru import logger

try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    AsyncTavilyClient = None
    TAVILY_AVAILABLE = False

from configs.config import settings

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class NewsArticle:
    """A news article from search results."""
    title: str
    url: str
    source: str
    snippet: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0


def _tavily_retry(func):
    """Apply retry decorator if tenacity is available."""
    if not TENACITY_AVAILABLE:
        return func
    return retry(
        stop=stop_after_attempt(settings.retry.max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry.min_wait_seconds,
            max=settings.retry.max_wait_seconds,
        ),
        reraise=True,
    )(func)


# =============================================================================
# SEARCH FUNCTION
# =============================================================================


@traceable(name="search_company_news", run_type="tool", tags=["search", "tavily"])
async def search_company_news(
    query: str,
    max_results: int = 5,
) -> list[NewsArticle]:
    """
    Search for company news using Tavily.

    Args:
        query: Search query (e.g., "Apple AAPL stock news")
        max_results: Maximum results to return

    Returns:
        List of NewsArticle objects
    """
    if not TAVILY_AVAILABLE or AsyncTavilyClient is None:
        logger.warning("Tavily not installed. Run: pip install tavily-python")
        return []

    if not settings.tavily.api_key:
        logger.warning("TAVILY_API_KEY not set")
        return []

    logger.info(f"Searching: {query}")

    @_tavily_retry
    async def _do_search():
        client = AsyncTavilyClient(api_key=settings.tavily.api_key)
        return await client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

    try:
        response = await _do_search()

        articles = []
        for result in response.get("results", []):
            articles.append(NewsArticle(
                title=result.get("title", ""),
                url=result.get("url", ""),
                source=_extract_source(result.get("url", "")),
                snippet=result.get("content", "")[:500],
                published_date=result.get("published_date"),
                relevance_score=result.get("score", 0.0),
            ))

        logger.info(f"Found {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def _extract_source(url: str) -> str:
    """Extract source name from URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        return domain.split(".")[0].title()
    except Exception:
        return "Unknown"

