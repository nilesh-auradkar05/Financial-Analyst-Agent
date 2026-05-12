from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path("evaluation/fixtures/rag_quality_eval_cases_v1.json")

def test_rag_quality_fixture_contract() -> None:
    cases = json.loads(FIXTURE.read_text())
    assert len(cases) == 12
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert {case["ticker"] for case in cases} == {"AAPL", "MSFT", "NVDA"}
    assert all(case["include_filing_analysis"] is True for case in cases)
    assert all(case["include_news_sentiment"] is False for case in cases)
    assert all(case["max_news_articles"] == 0 for case in cases)
    assert all(case["input"].endswith(".") for case in cases)
    assert all(len(case["expected_output"]) >= 80 for case in cases)
