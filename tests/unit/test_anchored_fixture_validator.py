"""S2-T02: anchored retrieval fixture validator (the "lie detector").

Traces to docs/test-plan.md §6 "Fixture validator" and
docs/retrieval-benchmark.md §§2, 3, 5, 6.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evaluation.validate_retrieval_fixture import (
    FixtureValidationError,
    load_anchored_fixture,
    validate_anchored_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SECTIONS = ["business", "risk_factors", "md&a", "market_risk"]
ACCESSIONS = {"AAPL": "0000320193-23-000106", "MSFT": "0000789019-23-000014"}
ANCHORS = {
    "business": "The company designs manufactures and markets smartphones personal computers tablets and wearables",
    "risk_factors": "Supply chain concentration in a few outsourcing partners could materially harm future operating results",
    "md&a": "Net sales increased during the fiscal year driven primarily by higher services revenue and iPhone demand",
    "market_risk": "A hypothetical ten percent change in interest rates would not have a material effect on fair value",
}


def _gold(
    ticker: str, section: str, anchor: str | None = None, relevance: int = 2
) -> dict[str, Any]:
    return {
        "accession_number": ACCESSIONS[ticker],
        "section_key": section,
        "anchor_text": ANCHORS[section] if anchor is None else anchor,
        "relevance": relevance,
    }


def _case(
    ticker: str, section: str, case_id: str | None = None, **overrides: Any
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": case_id or f"{ticker.lower()}-{section}-001",
        "ticker": ticker,
        "query": f"What does {ticker} disclose about {section}?",
        "filters": {
            "ticker": ticker,
            "filing_type": "10-K",
            "section_key": section,
            "filing_date_from": None,
            "filing_date_to": None,
            "accession_number": None,
        },
        "expected_sections": [section],
        "answer_intent": "section_specific",
        "gold_evidence": [_gold(ticker, section)],
        "expected_keywords": ["disclose"],
        "notes": None,
    }
    case.update(overrides)
    return case


def _fixture(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fixture_version": "retrieval_shared_benchmark_v1",
        "generated_at": "2026-06-03T00:00:00Z",
        "embedding_model": "test-embedding@1",
        "cases": cases,
    }


def _base_payload() -> dict[str, Any]:
    return _fixture([_case("AAPL", s) for s in SECTIONS])


def _write_sources(source_dir: Path, overrides: dict[tuple[str, str], str] | None = None) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for ticker, accession in ACCESSIONS.items():
        for section in SECTIONS:
            text = (overrides or {}).get(
                (ticker, section),
                f"Preamble for {section}. {ANCHORS[section]}. Closing remarks.",
            )
            (source_dir / f"{accession}_{section}.txt").write_text(text, encoding="utf-8")


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    path = tmp_path / "source_sections"
    _write_sources(path)
    return path


def _failures_for(payload: object, source_dir: Path, min_cases: int = 4) -> list[Any]:
    with pytest.raises(FixtureValidationError) as excinfo:
        validate_anchored_fixture(payload, source_dir=source_dir, min_cases=min_cases)
    return list(excinfo.value.failures)


def _assert_failure(
    payload: object,
    source_dir: Path,
    rule: str,
    case_id: str | None,
    min_cases: int = 4,
) -> Any:
    failures = _failures_for(payload, source_dir, min_cases)
    matches = [f for f in failures if f.rule == rule and f.case_id == case_id]
    assert matches, (
        f"expected rule={rule} case_id={case_id}, got {[(f.case_id, f.rule) for f in failures]}"
    )
    return matches[0]


# --- Valid fixture -----------------------------------------------------------


def test_valid_fixture_passes(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes."""
    fixture = validate_anchored_fixture(_base_payload(), source_dir=source_dir, min_cases=4)

    assert [c.case_id for c in fixture.cases] == [f"aapl-{s}-001" for s in SECTIONS]
    assert fixture.cases[0].gold_evidence[0].relevance == 2


def test_valid_fixture_loads_from_file(tmp_path: Path, source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes (file entry point)."""
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_base_payload()), encoding="utf-8")

    fixture = load_anchored_fixture(path, source_dir=source_dir, min_cases=4)

    assert len(fixture.cases) == 4


@pytest.mark.parametrize(
    "intent", ["factual", "comparison", "risk", "financial_performance", "section_specific"]
)
def test_every_allowed_answer_intent_passes(source_dir: Path, intent: str) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes (answer_intent enum)."""
    payload = _base_payload()
    payload["cases"][0]["answer_intent"] = intent

    validate_anchored_fixture(payload, source_dir=source_dir, min_cases=4)


def test_two_tickers_each_fully_covered_pass(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes (multi-ticker coverage)."""
    payload = _fixture([_case(t, s) for t in ACCESSIONS for s in SECTIONS])

    fixture = validate_anchored_fixture(payload, source_dir=source_dir, min_cases=8)

    assert len(fixture.cases) == 8


def test_anchor_of_exactly_eight_words_passes(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: anchor_text < 8 words (boundary: 8 passes)."""
    eight = " ".join(ANCHORS["business"].split()[:8])
    payload = _base_payload()
    payload["cases"][0]["gold_evidence"][0]["anchor_text"] = eight

    validate_anchored_fixture(payload, source_dir=source_dir, min_cases=4)


# --- Failure rules -----------------------------------------------------------


def test_duplicate_case_id_fails(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Duplicate case_id -> fails."""
    payload = _base_payload()
    payload["cases"].append(_case("AAPL", "business", case_id="aapl-business-001"))

    _assert_failure(payload, source_dir, "duplicate_case_id", "aapl-business-001", min_cases=4)


@pytest.mark.parametrize("variant", ["empty", "absent"])
def test_missing_gold_evidence_fails(source_dir: Path, variant: str) -> None:
    """test-plan §6 Fixture validator: Missing gold_evidence -> fails."""
    extra = _case("AAPL", "business", case_id="aapl-nogold-001")
    if variant == "empty":
        extra["gold_evidence"] = []
    else:
        del extra["gold_evidence"]
    payload = _base_payload()
    payload["cases"].append(extra)

    _assert_failure(payload, source_dir, "missing_gold", "aapl-nogold-001")


def test_unknown_section_key_fails(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Unknown section_key -> fails."""
    bad = _case("AAPL", "business", case_id="aapl-badsection-001")
    bad["gold_evidence"] = [_gold("AAPL", "business") | {"section_key": "legal_proceedings"}]
    bad["expected_sections"] = ["legal_proceedings"]
    payload = _base_payload()
    payload["cases"].append(bad)

    _assert_failure(payload, source_dir, "section_key", "aapl-badsection-001")


@pytest.mark.parametrize("relevance", [0, 3, -1])
def test_relevance_outside_one_two_fails(source_dir: Path, relevance: int) -> None:
    """test-plan §6 Fixture validator: relevance not in {1,2} -> fails."""
    bad = _case("AAPL", "business", case_id="aapl-badrel-001")
    bad["gold_evidence"] = [_gold("AAPL", "business", relevance=relevance)]
    payload = _base_payload()
    payload["cases"].append(bad)

    _assert_failure(payload, source_dir, "relevance", "aapl-badrel-001")


def test_anchor_under_eight_words_fails(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: anchor_text < 8 words -> fails."""
    seven = " ".join(ANCHORS["business"].split()[:7])
    bad = _case("AAPL", "business", case_id="aapl-short-001")
    bad["gold_evidence"] = [_gold("AAPL", "business", anchor=seven)]
    payload = _base_payload()
    payload["cases"].append(bad)

    _assert_failure(payload, source_dir, "anchor_length", "aapl-short-001")


def test_anchor_absent_from_source_fails_with_case_id_and_anchor(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Anchor absent from source cache -> fails with case_id + anchor."""
    invented = "Management invented this sentence which never appears anywhere in the filing text"
    payload = _base_payload()
    payload["cases"][1]["gold_evidence"][0]["anchor_text"] = invented

    failure = _assert_failure(payload, source_dir, "anchor_not_in_source", "aapl-risk_factors-001")

    assert invented in failure.detail


def test_anchor_with_missing_source_file_fails(tmp_path: Path) -> None:
    """test-plan §6 Fixture validator: Anchor absent from source cache (no cache file)."""
    empty_dir = tmp_path / "empty_cache"
    empty_dir.mkdir()

    failure = _assert_failure(
        _base_payload(), empty_dir, "anchor_not_in_source", "aapl-business-001"
    )

    assert ANCHORS["business"] in failure.detail


def test_expected_sections_mismatch_fails(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: expected_sections mismatch -> fails."""
    payload = _base_payload()
    payload["cases"][0]["expected_sections"] = ["business", "risk_factors"]

    _assert_failure(payload, source_dir, "expected_sections", "aapl-business-001")


def test_expected_sections_order_does_not_matter(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: expected_sections mismatch (set equality, not order)."""
    payload = _base_payload()
    case = payload["cases"][0]
    case["gold_evidence"].append(_gold("AAPL", "risk_factors"))
    case["expected_sections"] = ["risk_factors", "business"]

    validate_anchored_fixture(payload, source_dir=source_dir, min_cases=4)


def test_fixture_below_min_cases_fails(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Fixture below --min-cases -> fails."""
    _assert_failure(_base_payload(), source_dir, "min_cases", None, min_cases=5)


@pytest.mark.parametrize("ticker", ["aapl", "TOOLONG", "AAP1", "", "BRK.B"])
def test_invalid_ticker_format_fails(source_dir: Path, ticker: str) -> None:
    """test-plan §6 Fixture validator: Invalid ticker format -> fails."""
    bad = _case("AAPL", "business", case_id="bad-ticker-001")
    bad["ticker"] = ticker
    payload = _base_payload()
    payload["cases"].append(bad)

    _assert_failure(payload, source_dir, "ticker", "bad-ticker-001")


def test_answer_intent_outside_enum_fails(source_dir: Path) -> None:
    """retrieval-benchmark §5: answer_intent in allowed enum."""
    payload = _base_payload()
    payload["cases"][0]["answer_intent"] = "opinion"

    _assert_failure(payload, source_dir, "answer_intent", "aapl-business-001")


def test_ticker_missing_a_section_fails_coverage(source_dir: Path) -> None:
    """retrieval-benchmark §5: per-ticker and per-section coverage."""
    payload = _fixture(
        [_case("AAPL", s) for s in SECTIONS] + [_case("MSFT", s) for s in SECTIONS[:3]]
    )

    failure = _assert_failure(payload, source_dir, "coverage", None, min_cases=7)

    assert "MSFT" in failure.detail
    assert "market_risk" in failure.detail


@pytest.mark.parametrize(
    ("mutate", "rule"),
    [
        (lambda c: c.pop("query"), "schema"),
        (lambda c: c.update(unexpected_field=1), "schema"),
        (lambda c: c["gold_evidence"][0].update(unexpected_field=1), "schema"),
        (lambda c: c["gold_evidence"][0].update(relevance="high"), "schema"),
    ],
    ids=["missing_required", "extra_case_field", "extra_gold_field", "wrong_type"],
)
def test_schema_violations_fail_with_case_id(source_dir: Path, mutate: Any, rule: str) -> None:
    """test-plan §6 Fixture validator: Missing gold_evidence / shape errors -> fails (schema)."""
    payload = _base_payload()
    mutate(payload["cases"][0])

    _assert_failure(payload, source_dir, rule, "aapl-business-001")


def test_payload_missing_cases_fails_schema(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: malformed fixture -> fails (schema, fixture level)."""
    payload = _base_payload()
    del payload["cases"]

    _assert_failure(payload, source_dir, "schema", None)


def test_all_failures_are_collected_not_only_the_first(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: failures name every offending case_id + rule."""
    payload = _base_payload()
    payload["cases"][0]["answer_intent"] = "opinion"
    payload["cases"][2]["gold_evidence"][0]["anchor_text"] = (
        "Management invented this sentence which never appears anywhere in the filing text"
    )

    failures = _failures_for(payload, source_dir)
    pairs = {(f.case_id, f.rule) for f in failures}

    assert ("aapl-business-001", "answer_intent") in pairs
    assert ("aapl-md&a-001", "anchor_not_in_source") in pairs


def test_error_message_names_case_and_rule(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: fails with case_id (message is readable)."""
    payload = _base_payload()
    payload["cases"][0]["answer_intent"] = "opinion"

    with pytest.raises(FixtureValidationError) as excinfo:
        validate_anchored_fixture(payload, source_dir=source_dir, min_cases=4)

    assert "aapl-business-001" in str(excinfo.value)
    assert "answer_intent" in str(excinfo.value)


# --- Anchor matching uses norm() (retrieval-benchmark §3) ----------------------

_STRAIGHT = 'Management said "supply constraints" won\'t persist - and revenue grew 5% - during fiscal 2023 overall'


@pytest.mark.parametrize(
    ("anchor", "source"),
    [
        pytest.param(
            _STRAIGHT,
            "MD&A: Management said “supply constraints” won’t persist "
            "— and revenue grew 5% – during fiscal 2023 overall.",
            id="source_has_curly_quotes_and_long_dashes",
        ),
        pytest.param(
            "Management said “supply constraints” won’t persist "
            "— and revenue grew 5% – during fiscal 2023 overall",
            f"Intro. {_STRAIGHT}. Outro.",
            id="anchor_has_curly_quotes_and_long_dashes",
        ),
        pytest.param(
            _STRAIGHT.upper(),
            f"Intro. {_STRAIGHT}. Outro.",
            id="case_differs",
        ),
        pytest.param(
            _STRAIGHT.replace(" ", "  ").replace("grew", "grew\n\t"),
            f"Intro. {_STRAIGHT}. Outro.",
            id="anchor_whitespace_runs",
        ),
        pytest.param(
            _STRAIGHT,
            "Intro.\n\n" + _STRAIGHT.replace(" ", "\n  ", 4) + "\n\nOutro.",
            id="source_whitespace_runs",
        ),
        pytest.param(
            _STRAIGHT,
            f"Intro. {_STRAIGHT.replace('fiscal', 'ﬁscal').replace('during ', 'during ')}. Outro.",
            id="nfkc_ligature_and_nbsp",
        ),
    ],
)
def test_anchor_matching_uses_norm(tmp_path: Path, anchor: str, source: str) -> None:
    """retrieval-benchmark §3 norm(): NFKC, casefold, quote/dash mapping, whitespace collapse."""
    src = tmp_path / "src"
    _write_sources(src, {("AAPL", "md&a"): source})
    payload = _base_payload()
    payload["cases"][2]["gold_evidence"][0]["anchor_text"] = anchor

    validate_anchored_fixture(payload, source_dir=src, min_cases=4)


def test_anchor_differing_by_one_word_is_not_normalized_away(tmp_path: Path) -> None:
    """retrieval-benchmark §3: norm() forgives formatting, not content."""
    src = tmp_path / "src"
    _write_sources(src, {("AAPL", "md&a"): f"Intro. {_STRAIGHT}. Outro."})
    payload = _base_payload()
    payload["cases"][2]["gold_evidence"][0]["anchor_text"] = _STRAIGHT.replace("persist", "vanish")

    _assert_failure(payload, src, "anchor_not_in_source", "aapl-md&a-001")


def test_input_payload_is_not_mutated(source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes (validator is read-only)."""
    payload = _base_payload()
    before = copy.deepcopy(payload)

    validate_anchored_fixture(payload, source_dir=source_dir, min_cases=4)

    assert payload == before


# --- CLI ---------------------------------------------------------------------


def _run_cli(
    fixture_path: Path, source_dir: Path, min_cases: int = 4
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.validate_retrieval_fixture",
            str(fixture_path),
            "--source-dir",
            str(source_dir),
            "--min-cases",
            str(min_cases),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_cli_exits_zero_on_valid_anchored_fixture(tmp_path: Path, source_dir: Path) -> None:
    """test-plan §6 Fixture validator: Valid fixture -> passes (CLI)."""
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_base_payload()), encoding="utf-8")

    result = _run_cli(path, source_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AAPL" in result.stdout


def test_cli_exits_nonzero_and_prints_case_id_and_rule(tmp_path: Path, source_dir: Path) -> None:
    """test-plan §6 Fixture validator: fails with case_id (CLI, tab-separated lines)."""
    payload = _base_payload()
    payload["cases"][0]["answer_intent"] = "opinion"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli(path, source_dir)

    assert result.returncode == 1
    rows = [line.split("\t") for line in result.stdout.splitlines() if "\t" in line]
    assert any(row[:2] == ["aapl-business-001", "answer_intent"] for row in rows), result.stdout
