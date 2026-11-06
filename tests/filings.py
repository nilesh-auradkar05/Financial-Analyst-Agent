"""Shared filing builders for ingestion and OpenSearch tests."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

CRITICAL_SECTIONS = ["business", "risk_factors", "md&a", "market_risk"]


def make_filing(
    ticker: str,
    *,
    accession_number: str,
    filing_date: str,
    section_names: Iterable[str] | None = None,
) -> SimpleNamespace:
    names = list(
        section_names
        or [
            "Item 1. Business",
            "Item 1A. Risk Factors",
            "Item 7. Management's Discussion and Analysis",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        ]
    )
    sections = {
        name: SimpleNamespace(content=_section_text(ticker=ticker, section_name=name))
        for name in names
    }
    return SimpleNamespace(
        metadata=SimpleNamespace(
            ticker=ticker.lower(),
            filing_type="10-K",
            filing_date=filing_date,
            company_name=f"{ticker.upper()} Test Corp.",
            accession_number=accession_number,
            source_url=f"https://www.sec.gov/Archives/{accession_number}",
        ),
        sections=sections,
    )


def _section_text(*, ticker: str, section_name: str) -> str:
    return (
        f"{ticker.upper()} {section_name} disclosure describes operations, risks, "
        "liquidity, market exposure, controls, customers, products, and suppliers. " * 80
    )
