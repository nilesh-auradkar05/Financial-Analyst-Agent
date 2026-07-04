"""EdgarTools-based SEC 10-K section extraction."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger


@dataclass(frozen=True, slots=True)
class EdgarToolsSection:
    """A canonical 10-K section extracted by EdgarTools."""

    name: str
    item: str
    content: str

@dataclass(frozen=True, slots=True)
class EdgarToolsFilingPayload:
    """Latest 10-K payload extracted through EdgarTools."""

    ticker: str
    cik: str
    accession_number: str
    filing_type: str
    filing_date: str
    report_date: Optional[str]
    primary_document: str
    company_name: str
    sections: dict[str, EdgarToolsSection]

_SECTION_ITEMS: dict[str, str] = {
    "Business": "Item 1",
    "Risk Factors": "Item 1A",
    "MD&A": "Item 7",
    "Market Risk": "Item 7A",
}

_SECTION_PROPERTY_FALLBACKS: dict[str, tuple[str, ...]] = {
    "Business": ("business",),
    "Risk Factors": ("risk_factors",),
    "MD&A": (
        "management_discussion",
        "management_discussion_and_analysis",
        "mda",
    ),
    "Market Risk": (
        "market_risk",
        "quantitative_and_qualitative_disclosures",
    ),
}

_ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "Item 1": ("Item 1", "ITEM 1", "item 1", "1"),
    "Item 1A": ("Item 1A", "ITEM 1A", "item 1a", "1A", "1a"),
    "Item 7": ("Item 7", "ITEM 7", "item 7", "7"),
    "Item 7A": ("Item 7A", "ITEM 7A", "item 7a", "7A", "7a"),
}

_MIN_SECTION_CHARS: dict[str, int] = {
    "Business": 1_000,
    "Risk Factors": 1_000,
    "MD&A": 1_000,
    "Market Risk": 500,
}

_MARKDOWN_HEADING_PATTERNS: dict[str, re.Pattern[str]] = {
    "Business": re.compile(
        r"(?im)^\s#{0,6}\s*item\s+1\s*[\.\-:–—]?\s+business\b.*$"
    ),
    "Risk Factors": re.compile(
        r"(?im)^\s#{0,6}\s*item\s+1a\s*[\.\-:–—]?\s+risk\s+factors\b.*$"
    ),
    "MD&A": re.compile(
        r"(?im)^\s#{0,6}\s*item\s+7\s*[\.\-:–—]?\s+management'?s?\s+discussion\b.*$"
    ),
    "Market Risk": re.compile(
        r"(?im)^\s#{0,6}\s*item\s+7a\s*[\.\-:–—]?\s+quantitative\s+and\s+qualitative\b.*$"
    ),
}

_MARKDOWN_BOUNDARY_RE = re.compile(
    r"(?im)^\s#{0,6}\s*item\s+"
    r"(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9)"
    r"\s*[\.\-:–—]?\s+.*$"
)

def _require_edgartools() -> tuple[Any, Callable[[str], Any]]:
    try:
        edgar_module = import_module("edgar")
    except ImportError as exc:
        raise RuntimeError("edgartools is not installed") from exc

    company_cls = getattr(edgar_module, "Company", None)
    set_identity_fn = getattr(edgar_module, "set_identity", None)
    if company_cls is None or not callable(set_identity_fn):
        raise RuntimeError("edgartools does not expose Company and set_identity")

    return company_cls, set_identity_fn

def _configure_identity(identity: Optional[str]) -> Any:
    company_cls, set_identity_fn = _require_edgartools()

    resolved_identity = identity or os.getenv("EDGARTOOLS_IDENTITY")
    if not resolved_identity:
        logger.warning(
            "EDGAR_IDENTITY is not set. Set EDGAR_IDENTITY='Name email@example.com'"
        )
        return company_cls

    set_identity_fn(resolved_identity)
    return company_cls

def _clean_text(value: str) -> str:
    """Normalize extracted section text without destroying paragraphs."""
    value = re.sub(r"\x1b\[[0-9;]*m", "", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _stringify_section(raw: Any) -> str:
    """Convert an EdgarTools section object into plain text."""
    if raw is None:
        return ""

    for attr_name in ("text", "markdown", "to_text", "to_markdown"):
        attr = getattr(raw, attr_name, None)
        if attr is None:
            continue

        try:
            value = attr() if callable(attr) else attr
        except TypeError:
            continue
        except Exception as exc:
            logger.debug("Could not stringify section via %s: %s", attr_name, exc)
            continue

        if value:
            return _clean_text(str(value))

    return _clean_text(str(raw))


def _get_attr(obj: Any, names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return str(value)
    return default


def _extract_section_from_tenk(
    *,
    tenk: Any,
    section_name: str,
    item: str,
) -> str:
    """Extract one section from a TenK object using item access first."""
    aliases = _ITEM_ALIASES.get(item, (item,))

    for alias in aliases:
        try:
            raw = tenk[alias]
        except Exception as exc:
            logger.debug("TenK item access failed for %s/%s: %s", section_name, alias, exc)
            continue

        content = _stringify_section(raw)
        if len(content) >= _MIN_SECTION_CHARS[section_name]:
            return content

    for prop_name in _SECTION_PROPERTY_FALLBACKS.get(section_name, ()):
        try:
            raw = getattr(tenk, prop_name, None)
        except Exception as exc:
            logger.debug("TenK property access failed for %s.%s: %s", section_name, prop_name, exc)
            continue

        content = _stringify_section(raw)
        if len(content) >= _MIN_SECTION_CHARS[section_name]:
            return content

    return ""


def _extract_sections_from_markdown(markdown: str) -> dict[str, str]:
    """Fallback extraction from EdgarTools-generated Markdown.

    This is deliberately small. EdgarTools still does the heavy HTML-to-Markdown
    cleanup; this fallback only slices Item headings when direct TenK item access
    misses a section.
    """
    markdown = _clean_text(markdown)
    boundary_matches = list(_MARKDOWN_BOUNDARY_RE.finditer(markdown))
    extracted: dict[str, str] = {}

    for section_name, pattern in _MARKDOWN_HEADING_PATTERNS.items():
        candidates: list[str] = []

        for match in pattern.finditer(markdown):
            start = match.start()
            end = len(markdown)

            for boundary in boundary_matches:
                if boundary.start() > start:
                    end = boundary.start()
                    break

            content = markdown[start:end].strip()
            if len(content) >= _MIN_SECTION_CHARS[section_name]:
                candidates.append(content)

        if candidates:
            extracted[section_name] = max(candidates, key=len)

    return extracted


def _build_metadata_payload(
    *,
    ticker: str,
    filing: Any,
    sections: dict[str, EdgarToolsSection],
) -> EdgarToolsFilingPayload:
    cik = _get_attr(filing, ("cik",), default="")
    if cik.isdigit():
        cik = cik.zfill(10)

    accession_number = _get_attr(
        filing,
        ("accession_no", "accession_number", "accession"),
        default="",
    )

    primary_document = _get_attr(
        filing,
        ("primary_document",),
        default="",
    )

    company_name = _get_attr(
        filing,
        ("company", "company_name"),
        default=ticker.upper(),
    )

    filing_date = _get_attr(
        filing,
        ("filing_date",),
        default="",
    )

    report_date = _get_attr(
        filing,
        ("period_of_report", "report_date"),
        default="",
    )

    return EdgarToolsFilingPayload(
        ticker=ticker.upper(),
        cik=cik,
        accession_number=accession_number,
        filing_type="10-K",
        filing_date=filing_date,
        report_date=report_date or None,
        primary_document=primary_document,
        company_name=company_name,
        sections=sections,
    )


def extract_latest_10k_with_edgartools(
    ticker: str,
    *,
    identity: Optional[str] = None,
    record_evidence: bool = False,
    evidence_output_dir: Path | str | None = None,
) -> EdgarToolsFilingPayload:
    """Extract the latest 10-K sections for a ticker using EdgarTools.

    Raises:
        RuntimeError: if EdgarTools is not installed.
        ValueError: if no 10-K or no usable sections are found.
    """
    company_cls = _configure_identity(identity)

    normalized_ticker = ticker.upper().strip()
    company = company_cls(normalized_ticker)
    filing = company.get_filings(form="10-K").latest()

    if filing is None:
        raise ValueError(f"No latest 10-K found for {normalized_ticker}")

    tenk = filing.obj()
    if tenk is None:
        raise ValueError(f"EdgarTools could not build TenK object for {normalized_ticker}")

    sections: dict[str, EdgarToolsSection] = {}

    for section_name, item in _SECTION_ITEMS.items():
        content = _extract_section_from_tenk(
            tenk=tenk,
            section_name=section_name,
            item=item,
        )

        if content:
            sections[section_name] = EdgarToolsSection(
                name=section_name,
                item=item,
                content=content[:50_000],
            )

    missing = [name for name in _SECTION_ITEMS if name not in sections]
    if missing:
        logger.warning(
            "EdgarTools direct TenK extraction missed %s for %s; trying Markdown fallback",
            missing,
            normalized_ticker,
        )

        try:
            markdown = filing.markdown()
        except Exception as exc:
            logger.warning("EdgarTools markdown fallback failed for %s: %s", normalized_ticker, exc)
            markdown = ""

        if markdown:
            markdown_sections = _extract_sections_from_markdown(markdown)
            for section_name in missing:
                fallback_content = markdown_sections.get(section_name)
                if fallback_content:
                    sections[section_name] = EdgarToolsSection(
                        name=section_name,
                        item=_SECTION_ITEMS[section_name],
                        content=fallback_content[:50_000],
                    )

    if not sections:
        raise ValueError(f"EdgarTools extracted no usable sections for {normalized_ticker}")

    missing_after_fallback = [name for name in _SECTION_ITEMS if name not in sections]
    if missing_after_fallback:
        logger.warning(
            "EdgarTools extraction still missing %s for %s",
            missing_after_fallback,
            normalized_ticker,
        )

    payload = _build_metadata_payload(
        ticker=normalized_ticker,
        filing=filing,
        sections=sections,
    )
    if record_evidence:
        _record_sec_snapshot(payload, output_dir=evidence_output_dir)
    return payload


def _record_sec_snapshot(
    payload: EdgarToolsFilingPayload,
    *,
    output_dir: Path | str | None,
) -> None:
    from dataops.snapshot_writer import EvidenceSnapshotWriter

    writer = EvidenceSnapshotWriter(output_dir or Path("artifacts/dataops/evidence_snapshots"))
    writer.write_json(
        source_type="sec_filing",
        ticker=payload.ticker,
        natural_key=payload.accession_number,
        payload=asdict(payload),
        fetcher_name="edgartools_sec_extractor",
        fetcher_version="1",
        metadata={
            "filing_type": payload.filing_type,
            "filing_date": payload.filing_date,
            "report_date": payload.report_date or "",
            "primary_document": payload.primary_document,
        },
    )
