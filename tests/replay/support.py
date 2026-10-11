"""Shared helpers for the replay suite (docs/test-plan.md section 13).

Everything here is test infrastructure: the evidence fixture data, the release builder
that goes through the real dataops APIs, and a faithful in-process chat model.
Expected values are always computed from the fixture inputs, never typed in as answers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dataops.contracts import DatasetReleaseManifest, SourceType
from dataops.evidence_release import create_evidence_release
from dataops.registry import ReleaseRegistry
from dataops.snapshot_writer import EvidenceSnapshotWriter

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))

TICKERS = ("AAPL", "MSFT")
REQUIRED_SOURCE_TYPES: list[SourceType] = [
    "market_quote",
    "news_article",
    "sec_filing",
    "sentiment_score",
]
DEFAULT_RELEASE_NAME = "alpha-evidence"
DEFAULT_RELEASE_VERSION = "0.1.0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NetworkAccessError(RuntimeError):
    """Raised by the replay-suite socket guard on any network connect attempt."""


class LiveToolUsedError(AssertionError):
    """Raised by a fake that stands in for a live evidence path that replay must not touch."""


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and require that it carries a UTC offset."""
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, f"timestamp has no UTC offset: {value!r}"
    return parsed


@dataclass(frozen=True)
class SnapshotSpec:
    source_type: SourceType
    ticker: str
    natural_key: str
    payload: dict[str, Any]
    fetched_at: datetime


def build_specs(aapl_price: float = 220.5) -> list[SnapshotSpec]:
    """Snapshot inputs for two tickers. Fetch times are all different.

    MSFT holds the globally oldest fetch time on purpose, so an implementation that takes
    the oldest time across the whole release instead of per ticker is caught on AAPL.
    """
    aapl = [
        SnapshotSpec(
            "market_quote",
            "AAPL",
            "AAPL:2026-07-04T10:00:00+00:00",
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "current_price": aapl_price,
                "price_change_percent": 1.5,
                "market_cap": 3_400_000_000_000,
                "pe_ratio": 31.2,
                "sector": "Technology",
                "industry": "Consumer Electronics",
            },
            datetime(2026, 7, 4, 10, 0, tzinfo=UTC),
        ),
        # 11:45:30 at +05:30 is 06:15:30 UTC on 2026-07-01: the oldest AAPL fetch.
        SnapshotSpec(
            "news_article",
            "AAPL",
            "https://news.example.com/aapl/services",
            {
                "title": "Apple services revenue rises",
                "url": "https://news.example.com/aapl/services",
                "source": "Example News",
                "snippet": "Apple services revenue rose as subscriptions expanded.",
                "published_date": "2026-06-30",
            },
            datetime(2026, 7, 1, 11, 45, 30, tzinfo=IST),
        ),
        SnapshotSpec(
            "news_article",
            "AAPL",
            "https://news.example.com/aapl/ai",
            {
                "title": "Apple expands on-device AI features",
                "url": "https://news.example.com/aapl/ai",
                "source": "Wire Service",
                "snippet": "Apple expanded on-device AI features across its product line.",
                "published_date": "2026-07-02",
            },
            datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        ),
        SnapshotSpec(
            "sec_filing",
            "AAPL",
            "0000320193-26-000001",
            {
                "ticker": "AAPL",
                "accession_number": "0000320193-26-000001",
                "filing_type": "10-K",
                "filing_date": "2025-10-31",
                "sections": {
                    "Business": {
                        "name": "Business",
                        "item": "Item 1",
                        "content": "Apple designs and sells devices and services worldwide.",
                    },
                    "Risk Factors": {
                        "name": "Risk Factors",
                        "item": "Item 1A",
                        "content": "Apple depends on a global supply chain for its products.",
                    },
                },
            },
            datetime(2026, 7, 2, 9, 0, tzinfo=UTC),
        ),
    ]
    for offset, label in enumerate(["positive", "positive", "neutral"]):
        aapl.append(
            SnapshotSpec(
                "sentiment_score",
                "AAPL",
                f"fake-model:aapl:{offset}",
                {
                    "text": f"Apple sentiment sample {offset}.",
                    "label": label,
                    "confidence": 0.9 - offset * 0.1,
                    "scores": {"positive": 0.5, "negative": 0.2, "neutral": 0.3},
                    "model_name": "fake-model",
                },
                datetime(2026, 7, 3, 12, 5 + offset, tzinfo=UTC),
            )
        )

    msft = [
        SnapshotSpec(
            "market_quote",
            "MSFT",
            "MSFT:2026-06-28T10:00:00+00:00",
            {
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "current_price": 431.25,
                "price_change_percent": -0.4,
                "market_cap": 3_200_000_000_000,
                "pe_ratio": 35.8,
                "sector": "Technology",
                "industry": "Software",
            },
            datetime(2026, 6, 28, 10, 0, tzinfo=UTC),
        ),
        SnapshotSpec(
            "news_article",
            "MSFT",
            "https://news.example.com/msft/cloud",
            {
                "title": "Microsoft cloud demand stays strong",
                "url": "https://news.example.com/msft/cloud",
                "source": "Example News",
                "snippet": "Microsoft reported that cloud demand stayed strong this quarter.",
                "published_date": "2026-06-19",
            },
            datetime(2026, 6, 20, 8, 0, tzinfo=UTC),
        ),
        SnapshotSpec(
            "sec_filing",
            "MSFT",
            "0000789019-26-000002",
            {
                "ticker": "MSFT",
                "accession_number": "0000789019-26-000002",
                "filing_type": "10-K",
                "filing_date": "2025-07-30",
                "sections": {
                    "Business": {
                        "name": "Business",
                        "item": "Item 1",
                        "content": "Microsoft develops software, cloud services and devices.",
                    },
                },
            },
            datetime(2026, 6, 25, 9, 0, tzinfo=UTC),
        ),
        SnapshotSpec(
            "sentiment_score",
            "MSFT",
            "fake-model:msft:0",
            {
                "text": "Microsoft sentiment sample 0.",
                "label": "negative",
                "confidence": 0.7,
                "scores": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
                "model_name": "fake-model",
            },
            datetime(2026, 6, 26, 9, 0, tzinfo=UTC),
        ),
    ]
    return aapl + msft


@dataclass
class BuiltRelease:
    """A registered evidence_snapshot release plus the inputs it was built from."""

    release_id: str
    registry_root: Path
    snapshot_root: Path
    specs: list[SnapshotSpec]
    manifest: DatasetReleaseManifest
    _extra: dict[str, Any] = field(default_factory=dict)

    def specs_for(self, ticker: str, source_type: str | None = None) -> list[SnapshotSpec]:
        return [
            spec
            for spec in self.specs
            if spec.ticker == ticker and (source_type is None or spec.source_type == source_type)
        ]

    def oldest_fetched_at(self, ticker: str) -> datetime:
        return min(spec.fetched_at for spec in self.specs_for(ticker))

    def expected_news(self, ticker: str) -> list[dict[str, Any]]:
        fields = ("title", "url", "source", "snippet", "published_date")
        rows = [
            {key: spec.payload[key] for key in fields}
            for spec in self.specs_for(ticker, "news_article")
        ]
        return sorted(rows, key=lambda row: row["url"])

    def expected_stock(self, ticker: str) -> dict[str, Any]:
        quotes = self.specs_for(ticker, "market_quote")
        return dict(max(quotes, key=lambda spec: spec.fetched_at).payload)

    def expected_filing_chunks(self, ticker: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self.specs_for(ticker, "sec_filing"):
            for section in spec.payload["sections"].values():
                rows.append(
                    {
                        "text": section["content"],
                        "section": section["name"],
                        "filing_type": spec.payload["filing_type"],
                        "filing_date": spec.payload["filing_date"],
                    }
                )
        return sorted(rows, key=lambda row: (row["section"], row["text"]))

    def expected_sentiment(self, ticker: str) -> dict[str, Any]:
        counts = Counter(
            spec.payload["label"] for spec in self.specs_for(ticker, "sentiment_score")
        )
        if counts["positive"] > counts["negative"]:
            overall = "positive"
        elif counts["negative"] > counts["positive"]:
            overall = "negative"
        else:
            overall = "neutral"
        return {
            "overall_sentiment": overall,
            "positive_count": counts["positive"],
            "negative_count": counts["negative"],
            "neutral_count": counts["neutral"],
        }

    def quote_file(self, ticker: str) -> Path:
        return next((self.snapshot_root / "market_quote" / ticker).glob("*.json"))


def build_release(
    base: Path,
    *,
    name: str = DEFAULT_RELEASE_NAME,
    version: str = DEFAULT_RELEASE_VERSION,
    aapl_price: float = 220.5,
) -> BuiltRelease:
    """Write snapshots and register the release through the real dataops APIs."""
    snapshot_root = base / "snapshots"
    registry_root = base / "registry"
    specs = build_specs(aapl_price)
    writer = EvidenceSnapshotWriter(snapshot_root)
    for spec in specs:
        writer.write_json(
            source_type=spec.source_type,
            ticker=spec.ticker,
            natural_key=spec.natural_key,
            payload=spec.payload,
            fetcher_name=f"{spec.source_type}_fetcher",
            fetcher_version="1",
            fetched_at=spec.fetched_at,
        )
    manifest = create_evidence_release(
        dataset_name=name,
        dataset_version=version,
        snapshot_root=snapshot_root,
        registry_root=registry_root,
        tickers=list(TICKERS),
        required_source_types=REQUIRED_SOURCE_TYPES,
    )
    return BuiltRelease(
        release_id=manifest.release_id,
        registry_root=registry_root,
        snapshot_root=snapshot_root,
        specs=specs,
        manifest=manifest,
    )


def register_foreign_release(base: Path, release_id: str) -> Path:
    """Register a release of a type other than evidence_snapshot; returns the registry root."""
    name, version = release_id.split(":")
    registry_root = base / "foreign-registry"
    ReleaseRegistry(registry_root).append_release(
        DatasetReleaseManifest.create(
            dataset_name=name,
            dataset_version=version,
            release_type="quality_baseline",
            release_status="approved",
            snapshot_ids=[],
            config_hash="0" * 64,
            quality_report_uri=(base / "report.json").as_posix(),
            artifact_uri=(base / "artifact").as_posix(),
            parent_release_ids=[],
        )
    )
    return registry_root


class FakeChat:
    """Faithful chat model: writes one cited sentence per numbered source in its prompt.

    It computes its memo from the prompt it receives and opens no sockets. `prompts_seen`
    is its own public record of what it was asked, so a test can show it was never used.
    """

    _SOURCE = re.compile(r"\[(\d+)\] [^\n]+\n([^\n]+)")

    def __init__(self) -> None:
        self.prompts_seen: list[str] = []

    async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        prompt = messages[-1]["content"]
        self.prompts_seen.append(prompt)
        sources = self._SOURCE.findall(prompt)
        memo = "## Executive Summary\n" + "\n".join(f"{text} [{index}]" for index, text in sources)
        return SimpleNamespace(content=memo)


def evidence_view(state: dict[str, Any]) -> dict[str, Any]:
    """Project a run state onto the four evidence fields, in an order-independent shape."""
    return {
        "news_articles": sorted(
            (
                {
                    key: article.get(key)
                    for key in ("title", "url", "source", "snippet", "published_date")
                }
                for article in state.get("news_articles", [])
            ),
            key=lambda row: row["url"],
        ),
        "stock_data": dict(state.get("stock_data", {})),
        "filing_chunks": sorted(
            (
                {key: chunk.get(key) for key in ("text", "section", "filing_type", "filing_date")}
                for chunk in state.get("filing_chunks", [])
            ),
            key=lambda row: (row["section"], row["text"]),
        ),
        "sentiment_result": dict(state.get("sentiment_result", {})),
    }
