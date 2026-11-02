from __future__ import annotations

from typing import Any

import pytest

from app.verification.grounding import ClaimAssessment, evaluate_memo_grounding


def _filing(text: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "index": 1,
        "source_type": "sec_filing",
        "title": "Market Risk - 10-K",
        "text": text,
        "url": "https://www.sec.gov/Archives/edgar/data/example/filing.htm",
        "date": "2025-11-01",
        "chunk_id": "chunk-abc",
        "section": "Market Risk",
        "filing_type": "10-K",
    }
    entry.update(overrides)
    return entry


def _other(source_type: str, text: str) -> dict[str, Any]:
    if source_type == "news":
        return {
            "index": 1,
            "source_type": "news",
            "title": "Company posts results",
            "text": text,
            "url": "https://news.example.com/story",
            "date": "2025-11-02",
            "source": "Example Wire",
        }
    return {
        "index": 1,
        "source_type": source_type,
        "title": "Market Data (yfinance)",
        "text": text,
        "url": None,
        "date": None,
    }


def _only_claim(memo: str, evidence: list[Any]) -> ClaimAssessment:
    result = evaluate_memo_grounding(memo, evidence)
    assert len(result.claims) == 1
    return result.claims[0]


# Test-plan §7: Filing figure stated at a larger scale (VERIFY-SCALE)
@pytest.mark.parametrize(
    ("figure", "filing_value"),
    [
        pytest.param("$356.7 million", "356,711", id="thousands-to-million"),
        pytest.param("$3.33 billion", "3,333,751", id="thousands-to-billion"),
        pytest.param("$665.1 million", "665,140", id="thousands-to-million-2"),
    ],
)
def test_filing_thousands_figure_supports_larger_scale_claim(
    figure: str, filing_value: str
) -> None:
    memo = f"Net income was {figure} for the fiscal year [1]."
    evidence = [_filing(f"Net income {filing_value} for the fiscal year ended September.")]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is True
    assert claim.supported is True


# Test-plan §7: Filing figure stated at a larger scale (VERIFY-SCALE), figure in millions
@pytest.mark.parametrize("filing_value", ["4,152", "4,151.9"])
def test_filing_millions_figure_supports_billion_claim(filing_value: str) -> None:
    memo = "Total revenue was $4.2 billion for the fiscal year [1]."
    evidence = [_filing(f"Total revenue {filing_value} for the fiscal year ended September.")]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is True
    assert claim.supported is True


# Test-plan §7: Wrong scaled figure still fails
@pytest.mark.parametrize(
    ("figure", "scale_word", "filing_value"),
    [
        pytest.param("$356.9", "million", "356,711", id="million-off-by-two-tenths"),
        pytest.param("$3.35", "billion", "3,333,751", id="billion-off-by-two-hundredths"),
    ],
)
def test_wrong_scaled_figure_is_unsupported_and_named(
    figure: str, scale_word: str, filing_value: str
) -> None:
    memo = f"Net income was {figure} {scale_word} for the fiscal year [1]."
    evidence = [_filing(f"Net income {filing_value} for the fiscal year ended September.")]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is False
    assert claim.supported is False
    assert claim.reason is not None
    assert figure in claim.reason


# Test-plan §7: Scaling limited to filing body text (source type)
@pytest.mark.parametrize("source_type", ["news", "market_data", "sentiment"])
def test_scaling_does_not_apply_to_non_filing_entries(source_type: str) -> None:
    memo = "Net income was $356.7 million for the fiscal year [1]."
    evidence = [_other(source_type, "Net income 356,711 for the fiscal year ended September.")]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is False
    assert claim.supported is False


# Test-plan §7: Scaling limited to filing body text (identifiers, URL, date)
@pytest.mark.parametrize(
    ("claim_figure", "overrides"),
    [
        pytest.param(
            "$356.7 million",
            {"url": "https://www.sec.gov/Archives/edgar/data/356711/filing.htm"},
            id="url",
        ),
        pytest.param("$356.7 million", {"chunk_id": "chunk-356711-0003"}, id="chunk-id"),
        pytest.param("$2.0 million", {"date": "2026-03-31"}, id="date"),
    ],
)
def test_scaling_does_not_use_filing_identifiers(
    claim_figure: str, overrides: dict[str, Any]
) -> None:
    memo = f"Net income was {claim_figure} for the fiscal year [1]."
    evidence = [
        _filing("Net income was reported for the fiscal year ended September.", **overrides)
    ]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is False
    assert claim.supported is False


# Test-plan §7: Years never scale
@pytest.mark.parametrize(
    ("claim_figure", "filing_text"),
    [
        pytest.param(
            "$2.0 billion",
            "Net income rose in fiscal 2025 for the fiscal year.",
            id="year-millions",
        ),
        pytest.param(
            "$2.03 million",
            "Net income for fiscal 2026 for the fiscal year.",
            id="year-thousands",
        ),
    ],
)
def test_years_in_filing_text_do_not_scale(claim_figure: str, filing_text: str) -> None:
    memo = f"Net income was {claim_figure} for the fiscal year [1]."

    claim = _only_claim(memo, [_filing(filing_text)])

    assert claim.numbers_ok is False
    assert claim.supported is False


# Test-plan §7: Existing matches unchanged
@pytest.mark.parametrize(
    ("memo", "evidence"),
    [
        pytest.param(
            "Apple reported revenue of $394 billion for the year [1].",
            [_filing("Apple reported revenue of $394 billion for the year.")],
            id="same-scale-word",
        ),
        pytest.param(
            "Earnings per share rose to $37.32 on income growth [1].",
            [_filing("Earnings per share rose to 37.319225 on income growth.")],
            id="unscaled-rounded-at-stated-precision",
        ),
        pytest.param(
            "Gross margin was 12.5% on product sales [1].",
            [_filing("Gross margin was 12.5% on product sales.")],
            id="percentage",
        ),
        pytest.param(
            "Apple reported revenue of $394 billion for the year [1][2].",
            [
                _other("news", "Apple discussed the year with analysts."),
                _filing("Apple reported revenue of $394 billion for the year.", index=2),
            ],
            id="number-only-in-second-cited-entry",
        ),
    ],
)
def test_existing_number_matches_remain_supported(memo: str, evidence: list[Any]) -> None:
    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is True


# Test-plan §7: Filing figure stated at a larger scale (VERIFY-SCALE), per-figure judgement
def test_mixed_claim_names_only_the_unsupported_scaled_figure() -> None:
    memo = "Revenue grew from $3.22 billion to $9.99 billion over the fiscal year [1]."
    evidence = [_filing("Revenue grew over the fiscal year to 3,224,058 for the period.")]

    claim = _only_claim(memo, evidence)

    assert claim.numbers_ok is False
    assert claim.reason is not None
    assert "$9.99" in claim.reason
    assert "$3.22" not in claim.reason
