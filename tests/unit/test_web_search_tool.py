"""Behavior tests for the news evidence pipeline (`search_company_news`).

Traces to docs/test-plan.md §2:
  - "Empty retrieval | memo states limitation" — when every result is junk the
    tool must return no evidence (an honest empty list), not garbage.
  - "Citations | every memo citation resolves to returned evidence_id" — the
    snippets stored here become citable evidence, so they must be clean article
    text (no CAPTCHA/nav/markdown-image chrome) with real publication dates.

These assert observable outputs of the public `search_company_news` interface
against a fake Tavily client. They do not assert which internal method ran.
"""
from __future__ import annotations

import pytest

from app.services.tools import web_search_tool
from app.services.tools.web_search_tool import search_company_news


def _install_fake_tavily(monkeypatch, results, recorder):
    """Point the tool at a fake Tavily client that returns ``results``."""

    class _FakeTavilyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, **kwargs):
            recorder["kwargs"] = kwargs
            return {"results": results}

    monkeypatch.setattr(web_search_tool, "AsyncTavilyClient", _FakeTavilyClient)
    # Ensure the api-key guard passes without reading a real secret.
    monkeypatch.setattr(web_search_tool.settings.tavily, "api_key", "test-key")


def _good_article(title, url, content, *, score=0.8, date="Fri, 26 Jun 2026 04:17:07 GMT"):
    return {
        "title": title,
        "url": url,
        "content": content,
        "score": score,
        "published_date": date,
    }


# A representative mix mirroring the real Tavily garbage reproduced live.
_BOT_BLOCK = {
    "title": "Verify you are human",
    "url": "https://blocked.example.com/aapl",
    "content": "We have detected unusual traffic from your network. Are you a robot?",
    "score": 0.7,
    "published_date": None,
}
_EMPTY_QUOTE_PAGE = {
    "title": "Apple (AAPL) - Nasdaq",
    "url": "https://www.nasdaq.com/market-activity/stocks/aapl",
    "content": "Information Key data is currently not available. Please try other words.",
    "score": 0.6,
    "published_date": None,
}
_IMAGE_ONLY = {
    "title": "Apple quote - Robinhood",
    "url": "https://robinhood.com/us/en/stocks/AAPL",
    "content": "![Image 4: Apple logo](https://images.example.com/aapl.png)",
    "score": 0.5,
    "published_date": None,
}


@pytest.mark.asyncio
async def test_filters_junk_and_keeps_clean_articles(monkeypatch):
    recorder: dict = {}
    good1 = _good_article(
        "Apple hikes MacBook prices",
        "https://cnbc.com/2026/06/25/apple-price-hikes",
        "Apple stock slid after the company confirmed bigger Mac and iPad price "
        "hikes, its worst day in over a year.",
        score=0.72,
    )
    good2 = _good_article(
        "Micron surges on Apple memory pass-through",
        "https://finance.yahoo.com/markets/sandisk-micron",
        "Memory chip stocks rallied sharply after Apple confirmed it would pass "
        "through higher memory costs to consumers.",
        score=0.41,
        date="Thu, 18 Jun 2026 14:44:41 GMT",
    )
    results = [good1, _BOT_BLOCK, _EMPTY_QUOTE_PAGE, _IMAGE_ONLY, good2]
    _install_fake_tavily(monkeypatch, results, recorder)

    articles = await search_company_news("Apple Inc AAPL stock news", max_results=10)

    titles = [a.title for a in articles]
    assert titles == [good1["title"], good2["title"]]
    for a in articles:
        assert "![Image" not in a.snippet
        assert "unusual traffic" not in a.snippet.lower()
        assert "key data is currently not available" not in a.snippet.lower()
        assert a.published_date  # real dates survive


@pytest.mark.asyncio
async def test_snippet_is_cleaned_of_nav_and_images(monkeypatch):
    recorder: dict = {}
    content = (
        "* [Home](https://x.com)\n"
        "* [Markets](https://x.com/m)\n"
        "![Image 3: Apple logo](https://x.com/logo.png)\n"
        "Apple reported record Q2 revenue of $111.2 billion, up 17% from a year "
        "ago, beating the high end of guidance."
    )
    _install_fake_tavily(
        monkeypatch,
        [_good_article("Apple Q2 earnings", "https://news.example.com/q2", content)],
        recorder,
    )

    articles = await search_company_news("Apple AAPL", max_results=5)

    assert len(articles) == 1
    snippet = articles[0].snippet
    assert "Apple reported record Q2 revenue of $111.2 billion" in snippet
    assert "![Image" not in snippet
    assert "Image 3" not in snippet
    assert "[Home]" not in snippet
    assert "[Markets]" not in snippet


@pytest.mark.asyncio
async def test_all_junk_returns_empty(monkeypatch):
    recorder: dict = {}
    _install_fake_tavily(
        monkeypatch, [_BOT_BLOCK, _EMPTY_QUOTE_PAGE, _IMAGE_ONLY], recorder
    )

    articles = await search_company_news("Apple AAPL", max_results=10)

    assert articles == []


@pytest.mark.asyncio
async def test_dedup_by_url(monkeypatch):
    recorder: dict = {}
    url = "https://cnbc.com/2026/06/25/apple-price-hikes"
    content = (
        "Apple stock slid after the company confirmed bigger Mac and iPad price "
        "hikes, its worst day in over a year."
    )
    _install_fake_tavily(
        monkeypatch,
        [
            _good_article("Apple hikes prices", url, content),
            _good_article("Apple hikes prices (dupe)", url, content),
        ],
        recorder,
    )

    articles = await search_company_news("Apple AAPL", max_results=10)

    assert len(articles) == 1


@pytest.mark.asyncio
async def test_low_relevance_score_dropped(monkeypatch):
    recorder: dict = {}
    monkeypatch.setattr(web_search_tool.settings.tavily, "min_relevance_score", 0.3)
    body = (
        "Apple confirmed bigger Mac and iPad price hikes, sending the stock lower "
        "in its worst session in over a year."
    )
    keep = _good_article("Relevant", "https://news.example.com/keep", body, score=0.9)
    drop = _good_article("Weak match", "https://news.example.com/drop", body, score=0.1)
    _install_fake_tavily(monkeypatch, [keep, drop], recorder)

    articles = await search_company_news("Apple AAPL", max_results=10)

    assert [a.url for a in articles] == [keep["url"]]


@pytest.mark.asyncio
async def test_requests_recent_news_topic(monkeypatch):
    """The outbound request must ask Tavily for recent NEWS at the configured depth."""
    recorder: dict = {}
    body = (
        "Apple confirmed bigger Mac and iPad price hikes, sending the stock lower "
        "in its worst session in over a year."
    )
    _install_fake_tavily(
        monkeypatch,
        [_good_article("Apple news", "https://news.example.com/a", body)],
        recorder,
    )

    await search_company_news("Apple AAPL", max_results=7)

    kwargs = recorder["kwargs"]
    assert kwargs["topic"] == web_search_tool.settings.tavily.topic == "news"
    assert kwargs["search_depth"] == web_search_tool.settings.tavily.search_depth
    assert kwargs["max_results"] == 7
    assert kwargs["days"] == web_search_tool.settings.tavily.news_recency_days
