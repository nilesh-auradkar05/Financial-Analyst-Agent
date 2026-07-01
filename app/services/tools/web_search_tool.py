"""
WEB SEARCH TOOL (TAVILY)

This module provides web search capabilities using Tavily API.

Usage:
    from app.services.tools.web_search_tool import search_news, NewsArticle

    articles = await search_news("Apple earning Q2 2025")
    for article in articles:
        print(f"{article.title} ({article.source})")
        print(f"    URL: {article.url}")
        print(f"    Published: {article.published_date}")
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from langsmith import traceable
from loguru import logger
from tavily import AsyncTavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

# =============================================================================
# QUALITY FILTERING
# =============================================================================

# Markdown image syntax: ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# A line that consists solely of one or more markdown link bullets,
# e.g. "* [Text](url)." or "- [Text](url)"
_NAV_BULLET_LINE_RE = re.compile(
    r"^\s*[-*]\s*(\[[^\]]*\]\([^)]*\)[.,;:]?\s*)+$"
)

_WHITESPACE_RE = re.compile(r"\s+")

# Hard bot-block/empty-page markers. Presence of any of these means the
# scraped content is not a usable article and the result should be dropped.
_HARD_DROP_MARKERS: tuple[str, ...] = (
    "unusual traffic",
    "are you a robot",
    "not a robot",
    "captcha",
    "enable cookies",
    "access denied",
    "key data is currently not available",
    "data is currently not available",
)


def _clean_snippet(content: str) -> str:
    """Strip navigation chrome / markdown-image junk from raw Tavily content.

    - Removes markdown image syntax (``![alt](url)``).
    - Drops lines that are purely markdown link bullets (nav menus).
    - Collapses whitespace/newline runs into single spaces.
    """
    without_images = _MD_IMAGE_RE.sub("", content)

    kept_lines = [
        line
        for line in without_images.splitlines()
        if not _NAV_BULLET_LINE_RE.match(line)
    ]

    collapsed = _WHITESPACE_RE.sub(" ", " ".join(kept_lines))
    return collapsed.strip()


def _is_low_quality(cleaned: str, *, min_chars: int) -> bool:
    """Return True if the cleaned content should be dropped."""
    if len(cleaned) < min_chars:
        return True

    lowered = cleaned.lower()
    return any(marker in lowered for marker in _HARD_DROP_MARKERS)


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
    if max_results <= 0:
        logger.info(f"Skipping Tavily search because max_results={max_results}")
        return []

    if not settings.tavily.api_key:
        logger.warning("TAVILY_API_KEY not set")
        return []

    logger.info(f"Searching: {query}")

    @_tavily_retry
    async def _do_search() -> dict[str, Any]:
        client = AsyncTavilyClient(api_key=settings.tavily.api_key)
        search_kwargs: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "topic": settings.tavily.topic,
            "search_depth": settings.tavily.search_depth,
        }
        if settings.tavily.topic == "news":
            search_kwargs["days"] = settings.tavily.news_recency_days
        return await client.search(**search_kwargs)

    try:
        response = await _do_search()
        results = response.get("results", [])

        articles: list[NewsArticle] = []
        seen_urls: set[str] = set()
        for result in results:
            cleaned = _clean_snippet(result.get("content", ""))

            if result.get("score", 0.0) < settings.tavily.min_relevance_score:
                continue

            if _is_low_quality(cleaned, min_chars=settings.tavily.min_content_chars):
                continue

            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            articles.append(NewsArticle(
                title=result.get("title", ""),
                url=url,
                source=_extract_source(url),
                snippet=cleaned[:500],
                published_date=result.get("published_date"),
                relevance_score=result.get("score", 0.0),
            ))

        logger.info(
            f"Found {len(articles)} usable articles out of "
            f"{len(results)} returned by Tavily"
        )
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

