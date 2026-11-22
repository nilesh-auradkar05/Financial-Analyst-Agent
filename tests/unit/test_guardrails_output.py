"""Trace: docs/test-plan.md section 12, S7.1 rows "Output policy: price target", "Output policy:
guaranteed return", "Output policy: MNPI phrasing", "Output policy: uncited numeric claim",
"Output policy: PII" and "Disclaimer present" (SPEC 11.4).

`screen_memo` is fed what the graph feeds it: the sentences the real verifier reports as not
supported and as missing a citation (tests/unit/guardrail_support.py). The screen is
deterministic and makes no model or network call, so every expectation is on its return value.
New modules are imported inside the tests so each test fails on its own until they exist.
"""

from __future__ import annotations

import pytest

from tests.unit.guardrail_support import (
    codes,
    flagged_uncited,
    flagged_unsupported,
    registry_of,
    screened,
)

REVENUE = "Apple services revenue rose as subscriptions expanded."
MARGINS = "Margins rose in the quarter."
STIFEL = "Stifel lowered its price target to $400 on Monday."
REGISTRY = registry_of(REVENUE, MARGINS, STIFEL)

CITED_REVENUE = "Apple services revenue rose as subscriptions expanded [1]."
CITED_MARGINS = "Margins rose in the quarter [2]."
CITED_STIFEL = "Stifel lowered its price target to $400 on Monday [3]."


def memo_of(*lines: str) -> str:
    return "## 1. Executive Summary\n" + "\n".join(lines) + "\n"


def remaining_lines(memo: str) -> list[str]:
    return [line.strip() for line in memo.splitlines() if line.strip()]


# --- Output policy: price target -------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("Our price target is $250.", id="price-target"),
        pytest.param("Our target price is $250.", id="target-price"),
        pytest.param("Our price objective is $250.", id="price-objective"),
        pytest.param("OUR PRICE TARGET IS $250.", id="upper-case"),
    ],
)
def test_an_uncited_price_target_is_removed_and_recorded(sentence):
    """12 'Output policy: price target': a target with a number that the verifier does not mark
    supported (here: uncited) is removed; exactly one price_target_removed event."""
    memo = memo_of(CITED_REVENUE, sentence, CITED_MARGINS)
    assert flagged_uncited(memo, REGISTRY, "250"), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert "250" not in result
    assert CITED_REVENUE in result and CITED_MARGINS in result
    assert codes(events) == ["price_target_removed"]


def test_a_cited_price_target_the_source_does_not_hold_is_removed():
    """12 'Output policy: price target': cited, but the cited source does not support the figure
    -> the verifier does not mark it supported -> removed."""
    memo = memo_of(CITED_REVENUE, "Stifel lowered its price target to $400 on Monday [2].")
    assert flagged_unsupported(memo, REGISTRY, "$400"), "premise: [2] does not hold the figure"

    result, events = screened(memo, REGISTRY)

    assert "$400" not in result
    assert CITED_REVENUE in result
    assert codes(events) == ["price_target_removed"]


def test_a_cited_supported_third_party_target_is_kept():
    """12 'Output policy: price target': 'Stifel lowered its price target to $400 [N]' where the
    source holds it is kept; the memo comes back unchanged and no event is recorded."""
    memo = memo_of(CITED_REVENUE, CITED_STIFEL)
    assert not flagged_unsupported(memo, REGISTRY, "$400"), "premise: the source supports it"

    result, events = screened(memo, REGISTRY)

    assert result == memo
    assert events == []


def test_a_price_target_sentence_without_a_number_is_kept():
    """12 'Output policy: price target': the rule needs 'price target' plus a number."""
    sentence = "We do not publish a price target for revenue."
    memo = memo_of(CITED_REVENUE, sentence)
    assert flagged_uncited(memo, REGISTRY, "price target"), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert sentence in result
    assert events == []


# --- Output policy: guaranteed return ----------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("Holders enjoy a guaranteed return on this position.", id="guaranteed-return"),
        pytest.param(
            "The fund promises guaranteed profit for patient holders.", id="guaranteed-profit"
        ),
        pytest.param("This is a risk-free return for long holders.", id="risk-free-return"),
        pytest.param("With this position you cannot lose.", id="cannot-lose"),
        pytest.param("The bond is a GUARANTEED RETURN instrument.", id="upper-case"),
    ],
)
def test_guaranteed_return_language_is_removed_even_when_the_verifier_does_not_flag_it(sentence):
    """12 'Output policy: guaranteed return': removed whatever the verifier says; one
    guaranteed_return_removed event; the neighbouring sentences stay."""
    memo = memo_of(f"{CITED_REVENUE} {sentence} {CITED_MARGINS}")

    result, events = screened(memo, REGISTRY)

    assert sentence not in result
    assert CITED_REVENUE in result and CITED_MARGINS in result
    assert codes(events) == ["guaranteed_return_removed"]


def test_guaranteed_return_is_removed_from_a_recommendation_sentence():
    """12 'Output policy: uncited numeric claim' exemption: the recommendation sentence is not
    exempt from the guaranteed-return rule."""
    memo = memo_of(CITED_REVENUE, "We recommend a BUY rating for its guaranteed return.")

    result, events = screened(memo, REGISTRY)

    assert "guaranteed return" not in result
    assert codes(events) == ["guaranteed_return_removed"]


# --- Output policy: MNPI phrasing --------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("This view is based on insider information.", id="insider-information"),
        pytest.param("We hold material non-public information about the deal.", id="mnpi"),
        pytest.param(
            "The merger has not yet been publicly announced by the company.", id="unannounced"
        ),
        pytest.param("THE MERGER HAS NOT YET BEEN PUBLICLY ANNOUNCED.", id="upper-case"),
    ],
)
def test_mnpi_phrasing_is_removed_even_when_the_verifier_does_not_flag_it(sentence):
    """12 'Output policy: MNPI phrasing': removed whatever the verifier says; one mnpi_removed
    event; the neighbouring sentences stay."""
    memo = memo_of(f"{CITED_REVENUE} {sentence} {CITED_MARGINS}")

    result, events = screened(memo, REGISTRY)

    assert sentence not in result
    assert CITED_REVENUE in result and CITED_MARGINS in result
    assert codes(events) == ["mnpi_removed"]


# --- Output policy: uncited numeric claim ------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("Revenue was $12.5 billion in the quarter.", id="currency"),
        pytest.param("Operating margin was 14% last year.", id="percentage"),
        pytest.param(
            "The stock trades at 23.3x earnings, a revenue multiple above peers.", id="multiple"
        ),
        pytest.param("Revenue per share was 3.14 last year.", id="decimal"),
        pytest.param("Revenue reached 4 million units this year.", id="million"),
        pytest.param("Revenue reached 4 billion units this year.", id="billion"),
        pytest.param("Revenue reached 1.2 trillion yen this year.", id="trillion"),
        pytest.param(
            "The company expects to sell 5 million units, lifting revenue.", id="sell-then-number"
        ),
        pytest.param(
            "Institutional investors hold 62% of the revenue base.", id="hold-then-number"
        ),
    ],
)
def test_a_sentence_missing_a_citation_that_has_a_number_is_removed(sentence):
    """12 'Output policy: uncited numeric claim': uncited + number -> removed, one
    uncited_numeric_removed event; the cited neighbours stay."""
    memo = memo_of(CITED_REVENUE, sentence, CITED_MARGINS)
    assert flagged_uncited(memo, REGISTRY, sentence[:20]), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert sentence not in result
    assert CITED_REVENUE in result and CITED_MARGINS in result
    assert codes(events) == ["uncited_numeric_removed"]


def test_labels_that_look_numeric_are_not_numbers():
    """12 'Output policy: uncited numeric claim': '52-week', '12 months', '10-K' and 'Q3' are
    labels, not numbers: an uncited sentence holding only those is kept."""
    sentence = "Revenue stayed near its 52-week high over the last 12 months, per the 10-K and Q3 commentary."
    memo = memo_of(CITED_REVENUE, sentence)
    assert flagged_uncited(memo, REGISTRY, "52-week"), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert sentence in result
    assert events == []


def test_an_uncited_sentence_without_a_number_is_kept():
    """12 'Output policy: uncited numeric claim': the rule needs both the missing citation and a
    number; the verifier's other handling of uncited claims is not the screen's job."""
    sentence = "Revenue growth may continue next year."
    memo = memo_of(CITED_REVENUE, sentence)
    assert flagged_uncited(memo, REGISTRY, "may continue"), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert sentence in result
    assert events == []


def test_a_cited_numeric_sentence_is_not_removed_by_the_uncited_rule():
    """12 'Output policy: uncited numeric claim': a sentence that has a citation is not 'missing a
    citation', even when the cited source does not hold the figure (the verifier owns that)."""
    memo = memo_of(CITED_REVENUE, "Revenue was $500 billion [2].")
    assert flagged_unsupported(memo, REGISTRY, "$500"), "premise: cited but not supported"
    assert not flagged_uncited(memo, REGISTRY, "$500"), "premise: it does carry a citation"

    result, events = screened(memo, REGISTRY)

    assert result == memo
    assert events == []


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param(
            "We recommend a **HOLD** rating, as the stock trades at a P/E of 23.3x.",
            id="recommend-hold",
        ),
        pytest.param("We recommend investors review the 5% dividend yield.", id="recommend-only"),
        pytest.param(
            "A BUY recommendation is supported by 14% revenue growth.", id="buy-recommendation"
        ),
        pytest.param("We take a neutral stance with revenue up 12.5%.", id="neutral-stance"),
        pytest.param(
            "We maintain a SELL position given a 4.5% revenue decline.", id="sell-position"
        ),
        pytest.param("Our HOLD rating reflects revenue of $12.5 billion.", id="hold-rating"),
    ],
)
def test_the_recommendation_sentence_is_exempt_from_the_uncited_numeric_rule(sentence):
    """12 'Output policy: uncited numeric claim' (owner 2026-11-22): a sentence with 'recommend',
    or buy / hold / sell / neutral followed by rating, stance, position or recommendation, is
    kept even though it is uncited and numeric."""
    memo = memo_of(CITED_REVENUE, sentence)
    assert flagged_uncited(memo, REGISTRY, sentence[:20]), "premise: the verifier marks it uncited"

    result, events = screened(memo, REGISTRY)

    assert sentence in result
    assert events == []


def test_the_recommendation_sentence_is_not_exempt_from_the_price_target_rule():
    """12 'Output policy: uncited numeric claim': the exemption is not extended to the
    price-target rule."""
    memo = memo_of(CITED_REVENUE, "We recommend a BUY rating, and our price target is $300.")

    result, events = screened(memo, REGISTRY)

    assert "$300" not in result
    assert codes(events) == ["price_target_removed"]


def test_the_recommendation_sentence_is_not_exempt_from_pii_redaction():
    """12 'Output policy: uncited numeric claim': the exemption is not extended to the PII rule."""
    memo = memo_of(
        CITED_REVENUE, "We recommend a HOLD rating; write to ir@apple-ir.com for revenue questions."
    )

    result, events = screened(memo, REGISTRY)

    assert "ir@apple-ir.com" not in result
    assert "We recommend a HOLD rating; write to [redacted] for revenue questions." in result
    assert codes(events) == ["pii_redacted"]


@pytest.mark.parametrize("forced", ["uncited", "unsupported"])
def test_the_required_sec_filings_line_is_kept(forced):
    """12 'Output policy: uncited numeric claim' (section 16): `SEC Filings: Not Available.` is
    kept even when the verifier's sets name it."""
    from app.guardrails.output import screen_memo

    line = "SEC Filings: Not Available."
    memo = memo_of(CITED_REVENUE, line)
    flagged = {line}

    result, events = screen_memo(
        memo,
        flagged if forced == "unsupported" else set(),
        flagged if forced == "uncited" else set(),
    )

    assert line in result
    assert events == []


# --- Output policy: PII ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("ir@apple-ir.com", id="email"),
        pytest.param("Jane.Doe+news@example.co.uk", id="email-plus-and-subdomain"),
        pytest.param("(415) 555-0132", id="phone-parenthesised"),
        pytest.param("415-555-0132", id="phone-dashes"),
        pytest.param("415.555.0132", id="phone-dots"),
        pytest.param("123-45-6789", id="ssn"),
    ],
)
def test_pii_is_replaced_and_no_event_carries_the_value(value):
    """12 'Output policy: PII': an email, US phone number or US SSN becomes [redacted]; one
    pii_redacted event; the value is nowhere in the memo or the events."""
    sentence = f"Call the analyst desk at {value} for details."
    memo = memo_of(CITED_REVENUE, sentence)

    result, events = screened(memo, REGISTRY)

    assert value not in result
    assert "Call the analyst desk at [redacted] for details." in result
    assert CITED_REVENUE in result
    assert codes(events) == ["pii_redacted"]
    assert value not in repr(events)


def test_every_value_redacted_in_one_sentence_gets_its_own_event():
    """12 'Output policy: PII' / SPEC 11.4: one event per redaction."""
    memo = memo_of("Reach the desk at ir@apple-ir.com or 415-555-0132, ref 123-45-6789.")

    result, events = screened(memo, REGISTRY)

    assert result.count("[redacted]") == 3
    assert codes(events) == ["pii_redacted"] * 3
    for value in ("ir@apple-ir.com", "415-555-0132", "123-45-6789"):
        assert value not in repr(events)


def test_a_removed_sentence_does_not_leak_its_pii_through_the_event():
    """12 'Output policy: PII': no event carries the value, even when the sentence holding it is
    removed by another rule."""
    memo = memo_of(CITED_REVENUE, "Write to ir@apple-ir.com about the guaranteed return.")

    result, events = screened(memo, REGISTRY)

    assert "ir@apple-ir.com" not in result
    assert "ir@apple-ir.com" not in repr(events)
    assert "guaranteed_return_removed" in codes(events)


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("The filing is dated 2024-10-31 and the ticker is AAPL.", id="iso-date"),
        pytest.param("Item 1A starts on page 12 of the filing.", id="item-and-page"),
        pytest.param("Write to the desk at the address on file.", id="no-address"),
    ],
)
def test_text_that_is_not_pii_is_left_alone(sentence):
    """12 'Output policy: PII': dates and short numbers are not phone numbers or SSNs."""
    memo = memo_of(CITED_REVENUE, sentence)

    result, events = screened(memo, REGISTRY)

    assert sentence in result
    assert "[redacted]" not in result
    assert events == []


# --- The whole screen ---------------------------------------------------------------------------


def test_a_memo_with_every_defect_loses_exactly_the_defective_sentences():
    """12 'Output policy' rows together: one event per removed sentence or redaction, the cited
    supported target, the exempt recommendation and the cited claims all stay."""
    kept = [
        CITED_REVENUE,
        "We recommend a **HOLD** rating, as the stock trades at a P/E of 23.3x.",
        CITED_STIFEL,
        CITED_MARGINS,
    ]
    memo = (
        "## 1. Executive Summary\n"
        f"{kept[0]}\n"
        f"{kept[1]}\n"
        "Our price target is $250.\n"
        "Revenue was $12.5 billion in the quarter.\n"
        "The stock offers a guaranteed return for patient holders.\n"
        "The merger has not yet been publicly announced by the company.\n"
        "Call the analyst desk at 415-555-0132 for details.\n"
        "\n"
        "## 2. Outlook\n"
        f"{kept[2]}\n"
        f"{kept[3]}\n"
    )

    result, events = screened(memo, REGISTRY)

    assert codes(events) == sorted(
        [
            "price_target_removed",
            "uncited_numeric_removed",
            "guaranteed_return_removed",
            "mnpi_removed",
            "pii_redacted",
        ]
    )
    for line in kept:
        assert line in result
    assert "## 1. Executive Summary" in result and "## 2. Outlook" in result
    for gone in ("$250", "12.5 billion", "guaranteed return", "publicly announced", "415-555-0132"):
        assert gone not in result
    assert "Call the analyst desk at [redacted] for details." in result
    assert all(set(event) == {"code", "detail"} for event in events)


def test_a_clean_memo_comes_back_byte_for_byte_with_no_events():
    """12 'Guardrail events returned': a memo that triggers nothing records no events, and the
    screen does not reformat it."""
    memo = (
        "# Investment Memo: Apple Inc.\n\n"
        "## 1. Executive Summary\n"
        f"{CITED_REVENUE}\n\n"
        "## 2. Highlights\n"
        f"- **Margins:** {CITED_MARGINS}\n"
        f"- {CITED_STIFEL}\n"
        "- We recommend a **HOLD** rating while the evidence stays mixed.\n"
        "\n"
        "SEC Filings: Not Available.\n"
    )

    result, events = screened(memo, REGISTRY)

    assert result == memo
    assert events == []


def test_the_screen_is_deterministic():
    """12 intro: the same memo in gives the same memo and events out."""
    from app.guardrails.output import screen_memo

    from tests.unit.guardrail_support import verifier_flags

    memo = memo_of(
        CITED_REVENUE,
        "Our price target is $250.",
        "Revenue was $12.5 billion in the quarter.",
        "Write to ir@apple-ir.com about the guaranteed return.",
    )
    unsupported, uncited = verifier_flags(memo, REGISTRY)

    first = screen_memo(memo, unsupported, uncited)
    second = screen_memo(memo, set(unsupported), set(uncited))

    assert first == second


# --- Disclaimer present ------------------------------------------------------------------------

DISCLAIMER_TEXT = (
    "Disclaimer: This memo was generated automatically from public sources for information only. "
    "It is not investment advice or a recommendation to buy or sell any security, and it may "
    "contain errors. Do your own research or consult a licensed financial adviser."
)


def test_the_disclaimer_is_the_owner_approved_text():
    """12 'Disclaimer present': the fixed block, word for word (owner-approved 2026-11-22)."""
    from app.guardrails.output import DISCLAIMER

    assert DISCLAIMER == DISCLAIMER_TEXT


@pytest.mark.parametrize(
    "memo",
    [
        pytest.param("## Summary\nRevenue rose [1].", id="no-trailing-newline"),
        pytest.param("## Summary\nRevenue rose [1].\n", id="trailing-newline"),
        pytest.param("## Summary\nRevenue rose [1].\n\n\n", id="trailing-blank-lines"),
    ],
)
def test_the_memo_ends_with_the_disclaimer_exactly_once(memo):
    """12 'Disclaimer present': every returned memo ends with the block exactly once and keeps
    its own text in front of it."""
    from app.guardrails.output import DISCLAIMER, with_disclaimer

    result = with_disclaimer(memo)

    assert result.endswith(DISCLAIMER)
    assert result.count(DISCLAIMER) == 1
    assert result.startswith(memo.rstrip())


def test_adding_the_disclaimer_twice_changes_nothing():
    """12 'Disclaimer present': never twice, whichever path (draft or revise) added it."""
    from app.guardrails.output import DISCLAIMER, with_disclaimer

    once = with_disclaimer("## Summary\nRevenue rose [1].\n")
    twice = with_disclaimer(once)

    assert twice == once
    assert twice.count(DISCLAIMER) == 1
