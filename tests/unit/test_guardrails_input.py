"""Trace: docs/test-plan.md section 12, S7.1 rows "Ticker schema" and "Company name bounds"
(SPEC 11.4: ticker `^[A-Z]{1,6}(\\.[A-Z])?$`; company_name at most 80 printable ASCII characters,
no control characters).

The HTTP side of both rows (422 on /analyze, /analyze/async, /ingest and GET /ingest/{ticker}) is
in tests/test_api_integration.py; these tests pin the request models those routes validate with.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import AnalysisRequest, IngestionRequest

REQUEST_MODELS = [
    pytest.param(AnalysisRequest, id="analysis-request"),
    pytest.param(IngestionRequest, id="ingestion-request"),
]

PRINTABLE_ASCII = "".join(chr(code) for code in range(0x20, 0x7F))
EIGHTY_PRINTABLE = ("A" + PRINTABLE_ASCII[1:])[:80].strip()
assert len(EIGHTY_PRINTABLE) == 80


@pytest.mark.parametrize("model", REQUEST_MODELS)
@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        pytest.param("AAPL", "AAPL", id="four-letters"),
        pytest.param("A", "A", id="one-letter"),
        pytest.param("ABCDEF", "ABCDEF", id="six-letters"),
        pytest.param("BRK.B", "BRK.B", id="share-class"),
        pytest.param("ABCDEF.G", "ABCDEF.G", id="six-letters-and-share-class"),
        pytest.param("brk.b", "BRK.B", id="lower-case-is-normalized"),
        pytest.param("abcdef", "ABCDEF", id="six-lower-case-letters"),
    ],
)
def test_a_ticker_matching_the_schema_is_accepted(model, raw, normalized):
    """12 'Ticker schema': `BRK.B` and the 6-letter `ABCDEF` are accepted (and upper-cased)."""
    assert model(ticker=raw).ticker == normalized


@pytest.mark.parametrize("model", REQUEST_MODELS)
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("AAPL; DROP", id="injection"),
        pytest.param("ABCDEFG", id="seven-letters"),
        pytest.param("ABCDEFG.B", id="seven-letters-and-share-class"),
        pytest.param("BRK.BB", id="two-letter-share-class"),
        pytest.param("BRK.", id="dangling-dot"),
        pytest.param(".B", id="leading-dot"),
        pytest.param("A.B.C", id="two-dots"),
        pytest.param("BRK.1", id="digit-share-class"),
        pytest.param("ABC1", id="digit"),
        pytest.param("BRK-B", id="dash"),
        pytest.param("AA PL", id="inner-space"),
        pytest.param("", id="empty"),
    ],
)
def test_a_ticker_outside_the_schema_is_rejected(model, raw):
    """12 'Ticker schema': `AAPL; DROP` and the 7-letter `ABCDEFG` do not validate."""
    with pytest.raises(ValidationError) as raised:
        model(ticker=raw)

    assert any(error["loc"] == ("ticker",) for error in raised.value.errors())


def test_an_eighty_character_printable_ascii_company_name_is_accepted():
    """12 'Company name bounds': 80 characters spanning 0x20-0x7E are accepted unchanged."""
    assert (
        AnalysisRequest(ticker="AAPL", company_name=EIGHTY_PRINTABLE).company_name
        == EIGHTY_PRINTABLE
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("Apple Inc.", id="ordinary"),
        pytest.param("A B~", id="space-and-tilde-are-the-ends-of-the-range"),
        pytest.param("Berkshire Hathaway Inc. (Class B) - 10-K/A", id="punctuation"),
    ],
)
def test_a_printable_ascii_company_name_is_accepted(name):
    """12 'Company name bounds': printable ASCII (0x20-0x7E) is accepted."""
    assert AnalysisRequest(ticker="AAPL", company_name=name).company_name == name


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("A" * 81, id="eighty-one-characters"),
        pytest.param(EIGHTY_PRINTABLE + "x", id="eighty-one-printable"),
        pytest.param("Nestlé", id="non-ascii-letter"),
        pytest.param("Apple’s", id="typographic-apostrophe"),
        pytest.param("Acme \U0001f680", id="emoji"),
        pytest.param("ＡＰＰＬ", id="full-width-letters"),
        pytest.param("bad\nname", id="newline"),
        pytest.param("tab\tname", id="tab"),
        pytest.param("nul\x00name", id="nul"),
        pytest.param("esc\x1bname", id="escape"),
        pytest.param("del\x7fname", id="delete"),
        pytest.param("bell\x07name", id="bell"),
    ],
)
def test_a_company_name_that_is_too_long_or_not_printable_ascii_is_rejected(name):
    """12 'Company name bounds': 81 characters, control characters and non-ASCII are rejected."""
    with pytest.raises(ValidationError) as raised:
        AnalysisRequest(ticker="AAPL", company_name=name)

    assert any(error["loc"] == ("company_name",) for error in raised.value.errors())
