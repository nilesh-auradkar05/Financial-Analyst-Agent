from __future__ import annotations

from typing import Any

import pytest

from app.verification.grounding import ClaimAssessment, evaluate_memo_grounding

# Evidence url/date are None and titles are digit-free so the only digits the
# verifier can match come from the entry text under test.
_TITLES = {
    "sec_filing": "Management Discussion - Annual Report",
    "news": "Company posts results",
    "market_data": "Market Data (yfinance)",
    "sentiment": "Sentiment (FinBERT)",
}


def _entry(source_type: str, text: str) -> dict[str, Any]:
    return {
        "index": 1,
        "source_type": source_type,
        "title": _TITLES[source_type],
        "text": text,
        "url": None,
        "date": None,
    }


def _only_claim(memo: str, evidence: list[Any]) -> ClaimAssessment:
    result = evaluate_memo_grounding(memo, evidence)
    assert len(result.claims) == 1
    return result.claims[0]


# Test-plan §7: Scale words are part of the figure (VERIFY-UNITS), bare number
@pytest.mark.parametrize(
    ("source_type", "text"),
    [
        pytest.param(
            "sec_filing",
            "Net income is described in Item 7 of the annual report.",
            id="sec_filing-item-7",
        ),
        pytest.param("news", "Net income was reported in Item 7 of the filing.", id="news"),
        pytest.param("market_data", "Net income rank within the index: 7.", id="market_data"),
    ],
)
def test_bare_number_does_not_support_scaled_claim(source_type: str, text: str) -> None:
    claim = _only_claim(
        "Net income was $7.0 million for the year [1].", [_entry(source_type, text)]
    )

    assert claim.numbers_ok is False
    assert claim.reason is not None
    assert "$7.0" in claim.reason


# Test-plan §7: Scale words are part of the figure (VERIFY-UNITS), wrong magnitude
def test_wrong_magnitude_does_not_support_scaled_claim() -> None:
    evidence = [_entry("news", "Net income was reported at 7.0 billion for the year.")]

    claim = _only_claim("Net income was $7.0 million for the year [1].", evidence)

    assert claim.numbers_ok is False
    assert claim.reason is not None
    assert "$7.0" in claim.reason


# Test-plan §7: Scale words are part of the figure (VERIFY-UNITS), same absolute amount
@pytest.mark.parametrize(
    "figure",
    ["$7.0 million", "$7.0M", "7,000,000", "$7,000,000"],
)
def test_same_absolute_amount_supports_scaled_claim(figure: str) -> None:
    evidence = [_entry("news", f"Net income was reported at {figure} for the year.")]

    claim = _only_claim("Net income was $7.0 million for the year [1].", evidence)

    assert claim.numbers_ok is True


# Test-plan §7: Abbreviated scales in evidence (VERIFY-UNITS)
@pytest.mark.parametrize(
    ("claim_figure", "evidence_figure"),
    [
        pytest.param("$3.2 billion", "$3.2B", id="billion-B"),
        pytest.param("$450 million", "$450M", id="million-M"),
        pytest.param("$12 thousand", "$12K", id="thousand-K"),
        pytest.param("$3.2 billion", "3.2bn", id="billion-bn"),
        pytest.param("$4.25 trillion", "$4.25 trillion", id="trillion-word"),
    ],
)
def test_abbreviated_evidence_scale_supports_claim(claim_figure: str, evidence_figure: str) -> None:
    evidence = [_entry("news", f"Revenue came in at {evidence_figure} for the quarter.")]

    claim = _only_claim(f"Revenue was {claim_figure} for the quarter [1].", evidence)

    assert claim.numbers_ok is True


# Test-plan §7: Abbreviated scales in evidence (VERIFY-UNITS), market data text
def test_market_cap_humanized_text_supports_scaled_claim() -> None:
    evidence = [_entry("market_data", "Market capitalization: $90.00 billion ($90,000,000,000).")]

    claim = _only_claim("Market capitalization was $90.0 billion [1].", evidence)

    assert claim.numbers_ok is True


# Test-plan §7: Abbreviated scales in evidence (VERIFY-UNITS), abbreviation on the claim side
def test_claim_side_abbreviation_matches_written_out_scale() -> None:
    evidence = [_entry("news", "Revenue came in at 3.2 billion for the quarter.")]

    claim = _only_claim("Revenue was $3.2B for the quarter [1].", evidence)

    assert claim.numbers_ok is True


# Test-plan §7: Scale words are part of the figure (VERIFY-UNITS), claim-side abbreviation
def test_claim_side_abbreviation_not_supported_by_bare_number() -> None:
    evidence = [_entry("news", "Revenue growth was 3.2 in the quarter.")]

    claim = _only_claim("Revenue was $3.2B for the quarter [1].", evidence)

    assert claim.numbers_ok is False
    assert claim.reason is not None
    assert "$3.2" in claim.reason


# Test-plan §7: Unscaled claim figures unchanged (VERIFY-UNITS)
@pytest.mark.parametrize(
    ("source_type", "memo", "text"),
    [
        pytest.param(
            "market_data",
            "The current share price was $37.32 [1].",
            "Current share price: $37.319225.",
            id="price-rounding",
        ),
        pytest.param(
            "news",
            "Quarterly revenue growth was 12.5% [1].",
            "Quarterly revenue growth reached 12.5% year over year.",
            id="percent",
        ),
        pytest.param(
            "news",
            "Revenue grew strongly during fiscal 2025 [1].",
            "Revenue grew strongly during fiscal 2025 on higher demand.",
            id="year",
        ),
        pytest.param(
            "sentiment",
            "Negative articles were 8 of 9 analysed [1].",
            "Negative articles were 8 of 9 analysed.",
            id="count-of-total",
        ),
    ],
)
def test_unscaled_claim_figures_match_as_written(source_type: str, memo: str, text: str) -> None:
    claim = _only_claim(memo, [_entry(source_type, text)])

    assert claim.numbers_ok is True


# Test-plan §7: Existing matches unchanged (VERIFY-SCALE guard), filing table figure
def test_filing_comma_figure_still_supports_scaled_claim() -> None:
    evidence = [_entry("sec_filing", "Net income 356,711 for the fiscal year ended September.")]

    claim = _only_claim("Net income was $356.7 million for the fiscal year [1].", evidence)

    assert claim.numbers_ok is True
