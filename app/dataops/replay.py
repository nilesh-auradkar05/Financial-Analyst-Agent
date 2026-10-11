"""Runtime replay of a frozen evidence release (SPEC section 7.z).

Builds the seeded AgentState that a run drafts and verifies from when it replays a
registered ``evidence_snapshot`` release. No live tool, vector store or model is called.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from app.agents.state import AgentState, create_initial_state
from app.dataops.snapshot import freeze_evidence
from dataops.evidence_release import SnapshotRecord
from dataops.replay import load_evidence_release_records

RELEASE_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9._-]+:[A-Za-z0-9._-]+")
DEFAULT_REGISTRY_ROOT: Final = Path("artifacts/dataops")


def _company_name_from_records(ticker: str, records: list[SnapshotRecord]) -> str | None:
    for record in records:
        value = record.payload.get("company_name")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _news_articles_from_records(
    records: list[SnapshotRecord], max_news_articles: int
) -> list[dict]:
    articles: list[dict] = []
    for record in records:
        if record.snapshot.source_type != "news_article":
            continue
        payload = record.payload
        articles.append(
            {
                "title": str(payload.get("title") or "Untitled"),
                "url": str(payload.get("url") or record.snapshot.natural_key),
                "source": str(payload.get("source") or payload.get("publisher") or "Unknown"),
                "snippet": str(payload.get("snippet") or payload.get("content") or ""),
                "published_date": payload.get("published_date"),
            }
        )
    return articles[:max_news_articles]


def _stock_data_from_records(records: list[SnapshotRecord]) -> dict:
    market_records = [record for record in records if record.snapshot.source_type == "market_quote"]
    if not market_records:
        return {}
    latest = max(market_records, key=lambda record: record.snapshot.fetched_at)
    return dict(latest.payload)


def _filing_chunks_from_records(records: list[SnapshotRecord]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        if record.snapshot.source_type != "sec_filing":
            continue
        payload = record.payload
        accession_number = str(payload.get("accession_number") or record.snapshot.natural_key)
        sections = payload.get("sections")
        if not isinstance(sections, dict):
            continue
        for chunk_index, (section_name, section_payload) in enumerate(sections.items()):
            if isinstance(section_payload, dict):
                text = str(section_payload.get("content") or "")
                display_name = str(section_payload.get("name") or section_name)
            else:
                text = str(section_payload)
                display_name = str(section_name)
            if not text.strip():
                continue
            chunks.append(
                {
                    "text": text,
                    "section": display_name,
                    "filing_type": str(payload.get("filing_type") or "10-K"),
                    "filing_date": payload.get("filing_date"),
                    "relevance_score": 1.0,
                    "chunk_id": f"{accession_number}_{section_name}_{chunk_index:03d}",
                }
            )
    return chunks


def _sentiment_result_from_records(records: list[SnapshotRecord]) -> dict:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for record in records:
        if record.snapshot.source_type != "sentiment_score":
            continue
        label = str(record.payload.get("label") or "neutral").lower()
        if label not in counts:
            label = "neutral"
        counts[label] += 1

    if not any(counts.values()):
        return {}
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


def state_from_evidence_records(
    ticker: str,
    records: list[SnapshotRecord],
    *,
    include_filing_analysis: bool,
    include_news_sentiment: bool,
    max_news_articles: int,
) -> AgentState:
    """Return an initial AgentState seeded with the evidence in the snapshot records."""
    company_name = _company_name_from_records(ticker, records)
    state = create_initial_state(
        ticker,
        company_name,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )
    state["news_articles"] = _news_articles_from_records(records, max_news_articles)
    state["stock_data"] = _stock_data_from_records(records)
    if state["stock_data"].get("company_name"):
        state["company_name"] = str(state["stock_data"]["company_name"])
    if include_filing_analysis:
        state["filing_chunks"] = _filing_chunks_from_records(records)
    if include_news_sentiment:
        state["sentiment_result"] = _sentiment_result_from_records(records)
    return state


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def load_replay_state(
    release_id: str,
    ticker: str,
    *,
    registry_root: Path | str = DEFAULT_REGISTRY_ROOT,
    include_filing_analysis: bool = True,
    include_news_sentiment: bool = True,
    max_news_articles: int = 10,
) -> AgentState:
    """Return the seeded, frozen AgentState for one ticker of a registered evidence release."""
    if not isinstance(release_id, str) or RELEASE_ID_PATTERN.fullmatch(release_id) is None:
        raise ValueError(
            "release id must be formatted as <name:version> using letters, digits, '.', '_' or '-'"
        )
    bundle = load_evidence_release_records(release_id, registry_root=registry_root)
    symbol = ticker.upper()
    records = bundle.records_by_ticker.get(symbol)
    if not records:
        raise ValueError(f"evidence release {release_id} has no snapshots for ticker {symbol}")

    state = state_from_evidence_records(
        symbol,
        records,
        include_filing_analysis=include_filing_analysis,
        include_news_sentiment=include_news_sentiment,
        max_news_articles=max_news_articles,
    )
    as_of = min(_as_utc(record.snapshot.fetched_at) for record in records).isoformat()
    frozen = freeze_evidence(
        state,
        evidence_release=release_id,
        snapshot_ids=[record.snapshot.snapshot_id for record in records],
        as_of=as_of,
    )
    state["snapshot_hash"] = frozen["snapshot_hash"]
    state["evidence_as_of"] = frozen["evidence_as_of"]
    state["evidence_release"] = release_id
    return state
