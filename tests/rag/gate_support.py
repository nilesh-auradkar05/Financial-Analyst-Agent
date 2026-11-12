"""Shared builders for the CM-4 tests: a Gate-shaped corpus and its anchored fixture.

The corpus is one chunk per gold anchor, keyed by a semantic chunk ID, with the metadata the
ingestion pipeline writes.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.components.retrieval.vector_store import IndexDocument
from evaluation.validate_retrieval_fixture import AnchoredFixture, load_anchored_fixture

SECTIONS = ("business", "risk_factors", "md&a", "market_risk")
FILINGS = {
    "AAPL": ("Apple Inc.", "0000320193-24-000123", "2024-11-01"),
    "MSFT": ("Microsoft Corporation", "0000789019-24-000045", "2024-07-30"),
}
CHUNKS_PER_SECTION = 7  # 2 tickers x 4 sections x 7 = 56 cases, above the loader's 50-case floor


def anchor_for(ticker: str, section_key: str, index: int) -> str:
    return (
        f"{ticker} {section_key} passage {index} states that management reviews this "
        "exposure with the audit committee every quarter"
    )


def build_gate_chunks() -> list[IndexDocument]:
    chunks: list[IndexDocument] = []
    for ticker, (company, accession, filing_date) in FILINGS.items():
        for section_key in SECTIONS:
            for index in range(CHUNKS_PER_SECTION):
                chunk_id = f"{accession}_{section_key}_{index:03d}"
                text = (
                    f"Introductory filler for {company}. "
                    f"{anchor_for(ticker, section_key, index)}. Closing filler text."
                )
                chunks.append(
                    IndexDocument(
                        id=chunk_id,
                        text=text,
                        metadata={
                            "ticker": ticker,
                            "company_name": company,
                            "filing_type": "10-K",
                            "filing_date": filing_date,
                            "accession_number": accession,
                            "source_url": f"https://www.sec.gov/Archives/{accession}",
                            "section": section_key.replace("_", " ").title(),
                            "section_name": section_key.replace("_", " ").title(),
                            "section_key": section_key,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                        },
                    )
                )
    return chunks


def write_anchored_fixture(directory: Path) -> tuple[Path, Path]:
    """Write a valid anchored fixture and its source-section cache; return (fixture, source_dir)."""
    source_dir = directory / "source_sections"
    source_dir.mkdir()
    cases: list[dict[str, object]] = []
    for ticker, (_, accession, _) in FILINGS.items():
        for section_key in SECTIONS:
            anchors = [
                anchor_for(ticker, section_key, index) for index in range(CHUNKS_PER_SECTION)
            ]
            (source_dir / f"{accession}_{section_key}.txt").write_text(
                ". ".join(anchors) + ".", encoding="utf-8"
            )
            for index, anchor in enumerate(anchors):
                cases.append(
                    {
                        "case_id": f"{ticker.lower()}_{section_key.replace('&', '')}_{index}",
                        "ticker": ticker,
                        "query": f"What does {ticker} disclose in {section_key} part {index}?",
                        "filters": {"ticker": ticker, "filing_type": "10-K"},
                        "expected_sections": [section_key],
                        "answer_intent": "factual",
                        "gold_evidence": [
                            {
                                "accession_number": accession,
                                "section_key": section_key,
                                "anchor_text": anchor,
                                "relevance": 2,
                            }
                        ],
                        "expected_keywords": [],
                    }
                )
    payload = {
        "fixture_version": "cm4-test-1",
        "generated_at": "2026-11-08T00:00:00+00:00",
        "embedding_model": "bedrock:fake-hash-embed:16",
        "cases": cases,
    }
    fixture_path = directory / "anchored_fixture.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")
    return fixture_path, source_dir


def load_fixture_from(directory: Path) -> AnchoredFixture:
    fixture_path, source_dir = write_anchored_fixture(directory)
    return load_anchored_fixture(fixture_path, source_dir=source_dir)
