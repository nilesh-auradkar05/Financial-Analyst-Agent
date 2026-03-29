"""
Canonical SEC filing section definitions.

Single source of truth for section names, aliases, and normalization.
Imported by both `rag.ingestion` and `rag.vector_store`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalSection:
    """Immutable descriptor for a known SEC filing section."""

    key: str
    display_name: str
    slug: str


CANONICAL_SECTIONS: dict[str, CanonicalSection] = {
    "business": CanonicalSection(
        key="business",
        display_name="Business",
        slug="business",
    ),
    "risk_factors": CanonicalSection(
        key="risk_factors",
        display_name="Risk Factors",
        slug="risk_factors",
    ),
    "md&a": CanonicalSection(
        key="md&a",
        display_name="MD&A",
        slug="md_and_a",
    ),
    "market_risk": CanonicalSection(
        key="market_risk",
        display_name="Market Risk",
        slug="market_risk",
    ),
}

CANONICAL_SECTION_DISPLAY_NAMES: dict[str, str] = {
    key: sec.display_name for key, sec in CANONICAL_SECTIONS.items()
}

# Alias Lookup

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "business": (
        "business",
        "item 1",
        "item1",
        "company business",
    ),
    "risk_factors": (
        "risk factors",
        "risk factor",
        "item 1a",
        "item1a",
    ),
    "md&a": (
        "md&a",
        "management discussion and analysis",
        "managements discussion and analysis",
        "management discussion & analysis",
        "item 7",
        "item7",
    ),
    "market_risk": (
        "market risk",
        "quantitative and qualitative disclosures about market risk",
        "item 7a",
        "item7a",
    ),
}

DEFAULT_SECTIONS: list[str] = [
    "Business",
    "Risk Factors",
    "MD&A",
    "Market Risk",
]


# Normalization Helpers

def normalize_section_token(value: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    token = value.strip().lower()
    token = token.replace("&", " and ")
    token = token.replace("'", "").replace("\u2019", "")
    token = re.sub(r"[^a-z0-9]+", " ", token)
    return re.sub(r"\s+", " ", token).strip()

def canonical_section_key(section_name: str | None) -> str | None:
    """Map a raw section name to its canonical key, or ``None``."""
    if not section_name:
        return None
    normalized = normalize_section_token(section_name)
    for key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_token = normalize_section_token(alias)
            if normalized == alias_token or normalized.startswith(f"{alias_token} "):
                return key
    return None

def canonical_section_display_name(section_name: str | None) -> str | None:
    """Map a raw section name to its canonical display name, or ``None``."""
    key = canonical_section_key(section_name)
    if key is None:
        return None
    return CANONICAL_SECTION_DISPLAY_NAMES[key]

def canonicalize_section_name(section_name: str) -> CanonicalSection | None:
    """Return the full ``CanonicalSection`` for *section_name*, or ``None``."""
    key = canonical_section_key(section_name)
    if key is None:
        return None
    return CANONICAL_SECTIONS[key]

def fallback_section(section_name: str) -> CanonicalSection:
    """Create a non-canonical section descriptor for unknown section names."""
    slug = re.sub(r"[^a-z0-9]+", "_", section_name.strip().lower()).strip("_")
    key = slug or "unknown_section"
    return CanonicalSection(
        key=key,
        display_name=section_name.strip() or "Unknown Section",
        slug=key,
    )