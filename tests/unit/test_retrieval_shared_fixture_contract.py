from __future__ import annotations

from collections import Counter
from pathlib import Path

from evaluation.validate_retrieval_fixture import load_fixture

FIXTURE_PATH = Path("evaluation/fixtures/retrieval_shared_benchmark_v1.json")


def test_shared_retrieval_fixture_contract() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    cases = fixture.root

    assert 50 <= len(cases) <= 100
    assert len({case.id for case in cases}) == len(cases)
    assert {case.ticker for case in cases} == {"AAPL", "MSFT", "NVDA"}
    assert {case.mode for case in cases} == {"query"}

    ticker_counts = Counter(case.ticker for case in cases)
    assert ticker_counts == {"AAPL": 24, "MSFT": 24, "NVDA": 24}

    section_counts = Counter(section for case in cases for section in case.expected_sections)
    assert section_counts == {
        "business": 18,
        "risk_factors": 18,
        "md&a": 18,
        "market_risk": 18,
    }
