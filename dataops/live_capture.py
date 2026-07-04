from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataops.contracts import DatasetReleaseManifest, SourceType
from dataops.evidence_release import create_evidence_release

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA")
REQUIRED_EVIDENCE_SOURCE_TYPES: tuple[SourceType, ...] = (
    "sec_filing",
    "news_article",
    "market_quote",
    "sentiment_score",
)

SecExtractor = Callable[..., object]
AsyncEvidenceTool = Callable[..., Awaitable[Any]]
SentimentBatch = Callable[..., object]


@dataclass(frozen=True, slots=True)
class LiveEvidenceTools:
    sec_extractor: SecExtractor
    stock_fetcher: AsyncEvidenceTool
    news_search: AsyncEvidenceTool
    sentiment_batch: SentimentBatch


def _default_tools() -> LiveEvidenceTools:
    from app.services.sentiment import analyze_sentiment_batch
    from app.services.tools.edgartools_sec_extractor import extract_latest_10k_with_edgartools
    from app.services.tools.stock_data_tool import get_stock_data
    from app.services.tools.web_search_tool import search_company_news

    return LiveEvidenceTools(
        sec_extractor=extract_latest_10k_with_edgartools,
        stock_fetcher=get_stock_data,
        news_search=search_company_news,
        sentiment_batch=analyze_sentiment_batch,
    )


def _article_snippets(articles: Sequence[Any]) -> list[str]:
    snippets: list[str] = []
    for article in articles:
        if isinstance(article, dict):
            snippet = article.get("snippet")
        else:
            snippet = getattr(article, "snippet", None)
        if snippet:
            snippets.append(str(snippet))
    return snippets


async def record_evidence_release(
    *,
    dataset_name: str = "alpha-evidence",
    dataset_version: str,
    tickers: Sequence[str] = DEFAULT_TICKERS,
    snapshot_root: Path | str = Path("artifacts/dataops/evidence_snapshots"),
    registry_root: Path | str = Path("artifacts/dataops"),
    max_news_articles: int = 10,
    pointer_name: str = "evidence",
    tools: LiveEvidenceTools | None = None,
) -> DatasetReleaseManifest:
    resolved_tools = tools or _default_tools()
    output_dir = Path(snapshot_root)
    ticker_universe = [ticker.upper() for ticker in tickers]

    for ticker in ticker_universe:
        resolved_tools.sec_extractor(
            ticker,
            record_evidence=True,
            evidence_output_dir=output_dir,
        )
        await resolved_tools.stock_fetcher(
            ticker,
            record_evidence=True,
            evidence_output_dir=output_dir,
        )
        articles = await resolved_tools.news_search(
            f"{ticker} stock news",
            max_results=max_news_articles,
            record_evidence=True,
            evidence_ticker=ticker,
            evidence_output_dir=output_dir,
        )
        resolved_tools.sentiment_batch(
            _article_snippets(articles),
            record_evidence=True,
            evidence_ticker=ticker,
            evidence_output_dir=output_dir,
        )

    return create_evidence_release(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        snapshot_root=output_dir,
        registry_root=registry_root,
        tickers=ticker_universe,
        required_source_types=REQUIRED_EVIDENCE_SOURCE_TYPES,
        pointer_name=pointer_name,
    )


async def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record and register an evidence snapshot release.")
    parser.add_argument("--dataset-name", default="alpha-evidence")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--snapshot-root", default="artifacts/dataops/evidence_snapshots")
    parser.add_argument("--registry-root", default="artifacts/dataops")
    parser.add_argument("--max-news", type=int, default=10)
    parser.add_argument("--pointer-name", default="evidence")
    args = parser.parse_args(argv)

    manifest = await record_evidence_release(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        tickers=args.tickers,
        snapshot_root=args.snapshot_root,
        registry_root=args.registry_root,
        max_news_articles=args.max_news,
        pointer_name=args.pointer_name,
    )
    print(manifest.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
