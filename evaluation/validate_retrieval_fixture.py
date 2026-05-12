"""Validate retrieval benchmark fixtures before running expensive retrieval tests."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

SectionKey = Literal["business", "risk_factors", "md&a", "market_risk"]
RetrievalMode = Literal["query", "sections", "section_aware"]

ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
MIN_CASES = 50
MAX_CASES = 100


class RetrievalFixtureCase(BaseModel):
    """Single retrieval evaluation case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    ticker: str
    query: str
    filing_type: str | None = "10-K"
    expected_sections: list[SectionKey] = Field(min_length=1)
    expected_keywords: list[str] = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode = "query"
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("id must be lowercase snake_case with digits allowed")
        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.upper()
        if not re.fullmatch(r"[A-Z]{1,5}", ticker):
            raise ValueError("ticker must be 1-5 uppercase letters")
        return ticker

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if len(value.split()) < 5:
            raise ValueError("query is too short to be a realistic retrieval query")
        if not value.endswith("?"):
            raise ValueError("query should be phrased as a question")
        return value

    @field_validator("expected_keywords")
    @classmethod
    def validate_keywords(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if len({value.lower() for value in cleaned}) != len(cleaned):
            raise ValueError("expected_keywords must not contain duplicates")
        if any(len(value) < 2 for value in cleaned):
            raise ValueError("expected_keywords contains a value that is too short")
        return cleaned

    @model_validator(mode="after")
    def validate_case_id_prefix(self) -> RetrievalFixtureCase:
        expected_prefix = f"{self.ticker.lower()}_"
        if not self.id.startswith(expected_prefix):
            raise ValueError(f"id must start with ticker prefix {expected_prefix!r}")
        return self


class RetrievalFixture(RootModel[list[RetrievalFixtureCase]]):
    """Top-level retrieval fixture contract."""

    @model_validator(mode="after")
    def validate_fixture(self) -> RetrievalFixture:
        cases = self.root
        count = len(cases)
        if count < MIN_CASES or count > MAX_CASES:
            raise ValueError(f"fixture must contain {MIN_CASES}-{MAX_CASES} cases, got {count}")

        ids = [case.id for case in cases]
        duplicate_ids = sorted(case_id for case_id, seen in Counter(ids).items() if seen > 1)
        if duplicate_ids:
            raise ValueError(f"duplicate case ids found: {duplicate_ids}")

        ticker_section_counts: Counter[tuple[str, str]] = Counter(
            (case.ticker, section) for case in cases for section in case.expected_sections
        )
        tickers = {case.ticker for case in cases}
        required_sections: set[SectionKey] = {"business", "risk_factors", "md&a", "market_risk"}
        for ticker in tickers:
            missing_sections = sorted(
                section for section in required_sections if ticker_section_counts[(ticker, section)] == 0
            )
            if missing_sections:
                raise ValueError(f"ticker {ticker} is missing sections: {missing_sections}")

        return self


def load_fixture(path: Path) -> RetrievalFixture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RetrievalFixture.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a retrieval fixture JSON file.")
    parser.add_argument("fixture", type=Path, help="Path to retrieval fixture JSON file.")
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    ticker_counts = Counter(case.ticker for case in fixture.root)
    section_counts = Counter(section for case in fixture.root for section in case.expected_sections)

    print(f"Validated {len(fixture.root)} retrieval cases from {args.fixture}")
    print("Ticker distribution:")
    for ticker, count in sorted(ticker_counts.items()):
        print(f"  {ticker}: {count}")
    print("Section distribution:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count}")


if __name__ == "__main__":
    main()
