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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langsmith import traceable
from loguru import logger

try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    AsyncTavilyClient = None
    TAVILY_AVAILABLE = False

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

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

    try:
        client = AsyncTavilyClient(api_key=settings.tavily.api_key)

        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

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


# =============================================================================
# CLI
# =============================================================================


async def _main():
    """Test search."""
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Apple AAPL stock news"

    print(f"\nSearching: {query}\n")

    articles = await search_company_news(query)

    if articles:
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.title}")
            print(f"   Source: {article.source} | URL: {article.url}")
            print(f"   {article.snippet[:100]}...")
            print()
    else:
        print("No results found")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
