"""Validate retrieval benchmark fixtures before running expensive retrieval tests."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

SectionKey = Literal["business", "risk_factors", "md&a", "market_risk"]
RetrievalMode = Literal["query", "sections", "section_aware"]

ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
MIN_CASES = 50
MAX_CASES = 100

SECTION_KEYS = frozenset(get_args(SectionKey))
ANSWER_INTENTS = frozenset(
    {"factual", "comparison", "risk", "financial_performance", "section_specific"}
)
TICKER_PATTERN = re.compile(r"[A-Z]{1,5}")
MIN_ANCHOR_WORDS = 8
FILTER_KEYS = frozenset(
    {
        "ticker",
        "filing_type",
        "section_key",
        "filing_date_from",
        "filing_date_to",
        "accession_number",
    }
)
_NORM_TABLE = str.maketrans(
    {
        **dict.fromkeys("‘’‚‛", "'"),
        **dict.fromkeys("“”„", '"'),
        **dict.fromkeys("–—‒―−", "-"),
    }
)


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


def normalize_text(text: str) -> str:
    """Anchor/source normalization from retrieval-benchmark §3 ``norm()``."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_NORM_TABLE)
    return " ".join(folded.split())


@dataclass(frozen=True)
class FixtureFailure:
    case_id: str | None
    rule: str
    detail: str


class FixtureValidationError(ValueError):
    def __init__(self, failures: list[FixtureFailure]) -> None:
        self.failures = failures
        super().__init__("\n".join(f"{f.case_id or '-'} [{f.rule}] {f.detail}" for f in failures))


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accession_number: str
    section_key: str
    anchor_text: str
    relevance: StrictInt


class AnchoredCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    ticker: str
    query: str
    filters: dict[str, str | None]
    expected_sections: list[str]
    answer_intent: str
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("filters")
    @classmethod
    def validate_filter_keys(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        unknown = sorted(set(value) - FILTER_KEYS)
        if unknown:
            raise ValueError(f"unknown filter keys: {unknown}")
        return value


class AnchoredFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_version: str
    generated_at: str
    embedding_model: str
    cases: list[AnchoredCase]


def _schema_failure(case_id: str | None, error: ValidationError) -> FixtureFailure:
    detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return FixtureFailure(case_id, "schema", detail)


def _source_text(source_dir: Path, accession: str, section_key: str) -> str | None:
    path = source_dir / f"{accession}_{section_key}.txt"
    if path.parent.resolve() != source_dir.resolve() or not path.is_file():
        return None
    return normalize_text(path.read_text(encoding="utf-8"))


def _case_failures(case: AnchoredCase, source_dir: Path) -> list[FixtureFailure]:
    failures: list[FixtureFailure] = []

    def fail(rule: str, detail: str) -> None:
        failures.append(FixtureFailure(case.case_id, rule, detail))

    if not TICKER_PATTERN.fullmatch(case.ticker):
        fail("ticker", f"ticker {case.ticker!r} must be 1-5 uppercase letters")
    if case.answer_intent not in ANSWER_INTENTS:
        fail("answer_intent", f"answer_intent {case.answer_intent!r} is not allowed")
    if not case.gold_evidence:
        fail("missing_gold", "gold_evidence is empty")
    if set(case.expected_sections) != {g.section_key for g in case.gold_evidence}:
        fail("expected_sections", "expected_sections must equal the set of gold section_keys")
    for gold in case.gold_evidence:
        if gold.relevance not in (1, 2):
            fail("relevance", f"relevance {gold.relevance} must be 1 or 2")
        if len(gold.anchor_text.split()) < MIN_ANCHOR_WORDS:
            fail(
                "anchor_length",
                f"anchor has fewer than {MIN_ANCHOR_WORDS} words: {gold.anchor_text}",
            )
        if gold.section_key not in SECTION_KEYS:
            fail("section_key", f"section_key {gold.section_key!r} is not a canonical section")
            continue
        source = _source_text(source_dir, gold.accession_number, gold.section_key)
        if source is None or normalize_text(gold.anchor_text) not in source:
            fail(
                "anchor_not_in_source",
                f"anchor not found in {gold.accession_number}_{gold.section_key}: "
                f"{gold.anchor_text}",
            )
    return failures


def validate_anchored_fixture(
    payload: object, *, source_dir: Path, min_cases: int = 50
) -> AnchoredFixture:
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(raw_cases, list):
        raise FixtureValidationError(
            [FixtureFailure(None, "schema", "payload must be an object with a 'cases' list")]
        )

    failures: list[FixtureFailure] = []
    try:
        AnchoredFixture.model_validate({**payload, "cases": []})
    except ValidationError as error:
        failures.append(_schema_failure(None, error))

    cases: list[AnchoredCase] = []
    for raw in raw_cases:
        try:
            case = AnchoredCase.model_validate(raw)
        except ValidationError as error:
            raw_id = raw.get("case_id") if isinstance(raw, dict) else None
            failures.append(_schema_failure(raw_id if isinstance(raw_id, str) else None, error))
            continue
        cases.append(case)
        failures.extend(_case_failures(case, source_dir))

    id_counts = Counter(case.case_id for case in cases)
    failures.extend(
        FixtureFailure(case_id, "duplicate_case_id", f"case_id appears {count} times")
        for case_id, count in id_counts.items()
        if count > 1
    )
    if len(raw_cases) < min_cases:
        failures.append(
            FixtureFailure(
                None, "min_cases", f"fixture has {len(raw_cases)} cases, need {min_cases}"
            )
        )
    covered = {(case.ticker, g.section_key) for case in cases for g in case.gold_evidence}
    for ticker in sorted({case.ticker for case in cases}):
        missing = sorted(s for s in SECTION_KEYS if (ticker, s) not in covered)
        if missing:
            failures.append(
                FixtureFailure(None, "coverage", f"ticker {ticker} lacks gold anchors in {missing}")
            )

    if failures:
        raise FixtureValidationError(failures)
    return AnchoredFixture.model_validate(payload)


def load_anchored_fixture(path: Path, *, source_dir: Path, min_cases: int = 50) -> AnchoredFixture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_anchored_fixture(payload, source_dir=source_dir, min_cases=min_cases)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a retrieval fixture JSON file.")
    parser.add_argument("fixture", type=Path, help="Path to retrieval fixture JSON file.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("evaluation/fixtures/source_sections"),
        help="Source-section cache used to verify anchored fixtures.",
    )
    parser.add_argument("--min-cases", type=int, default=MIN_CASES)
    args = parser.parse_args()

    payload = json.loads(args.fixture.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        fixture = RetrievalFixture.model_validate(payload)
        ticker_counts = Counter(case.ticker for case in fixture.root)
        section_counts = Counter(
            section for case in fixture.root for section in case.expected_sections
        )

        print(f"Validated {len(fixture.root)} retrieval cases from {args.fixture}")
        print("Ticker distribution:")
        for ticker, count in sorted(ticker_counts.items()):
            print(f"  {ticker}: {count}")
        print("Section distribution:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count}")
        return 0

    try:
        anchored = validate_anchored_fixture(
            payload, source_dir=args.source_dir, min_cases=args.min_cases
        )
    except FixtureValidationError as error:
        for failure in error.failures:
            print(f"{failure.case_id or '-'}\t{failure.rule}\t{' '.join(failure.detail.split())}")
        return 1

    anchor_counts = Counter(
        (case.ticker, gold.section_key) for case in anchored.cases for gold in case.gold_evidence
    )
    print(f"Validated {len(anchored.cases)} anchored cases from {args.fixture}")
    print("Anchors per ticker and section:")
    for (ticker, section_key), count in sorted(anchor_counts.items()):
        print(f"  {ticker} {section_key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
