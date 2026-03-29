"""Tests for rag.sections — the shared canonical section registry.

This module was extracted (item 11) so ingestion and vector_store share
one source of truth.  These tests verify normalization, alias resolution,
and fallback behavior.
"""

from rag.sections import (
    CANONICAL_SECTION_DISPLAY_NAMES,
    CANONICAL_SECTIONS,
    DEFAULT_SECTIONS,
    SECTION_ALIASES,
    CanonicalSection,
    canonical_section_display_name,
    canonical_section_key,
    canonicalize_section_name,
    fallback_section,
    normalize_section_token,
)

# Easy: basic lookups

class TestNormalization:
    def test_lowercase_and_strip(self):
        assert normalize_section_token("  Business  ") == "business"

    def test_ampersand_expansion(self):
        assert normalize_section_token("MD&A") == "md and a"

    def test_apostrophe_removal(self):
        assert normalize_section_token("Management's") == "managements"

    def test_special_chars_to_space(self):
        assert normalize_section_token("Item-1A. Risk") == "item 1a risk"


class TestCanonicalKeyLookup:
    """Medium: alias mapping."""

    def test_exact_match(self):
        assert canonical_section_key("business") == "business"

    def test_display_name_match(self):
        assert canonical_section_key("Risk Factors") == "risk_factors"

    def test_item_number_match(self):
        assert canonical_section_key("Item 1A") == "risk_factors"
        assert canonical_section_key("Item 7") == "md&a"
        assert canonical_section_key("Item 7A") == "market_risk"

    def test_mda_variants(self):
        assert canonical_section_key("Management Discussion and Analysis") == "md&a"
        assert canonical_section_key("Management's Discussion and Analysis") == "md&a"
        assert canonical_section_key("Management Discussion & Analysis") == "md&a"

    def test_unknown_section_returns_none(self):
        assert canonical_section_key("Executive Compensation") is None
        assert canonical_section_key("") is None
        assert canonical_section_key(None) is None


class TestCanonicalDisplayName:
    def test_known_section(self):
        assert canonical_section_display_name("item 1a") == "Risk Factors"

    def test_unknown_returns_none(self):
        assert canonical_section_display_name("Something Unknown") is None


class TestCanonicalize:
    """Hard: full CanonicalSection resolution."""

    def test_returns_canonical_section_object(self):
        result = canonicalize_section_name("Risk Factors")
        assert isinstance(result, CanonicalSection)
        assert result.key == "risk_factors"
        assert result.display_name == "Risk Factors"
        assert result.slug == "risk_factors"

    def test_md_and_a_slug(self):
        result = canonicalize_section_name("MD&A")
        assert result is not None
        assert result.slug == "md_and_a"

    def test_unknown_returns_none(self):
        assert canonicalize_section_name("Random Heading") is None


class TestFallback:
    """Edge: fallback for non-canonical sections."""

    def test_fallback_creates_slug(self):
        fb = fallback_section("Executive Compensation")
        assert fb.key == "executive_compensation"
        assert fb.display_name == "Executive Compensation"

    def test_fallback_empty_string(self):
        fb = fallback_section("")
        assert fb.key == "unknown_section"
        assert fb.display_name == "Unknown Section"

    def test_fallback_special_chars(self):
        fb = fallback_section("Item 5.  Other Info!")
        assert fb.key == "item_5_other_info"


class TestRegistryConsistency:
    """Verify CANONICAL_SECTIONS and SECTION_ALIASES stay in sync."""

    def test_every_alias_key_has_canonical(self):
        for key in SECTION_ALIASES:
            assert key in CANONICAL_SECTIONS, f"Alias key {key!r} missing from CANONICAL_SECTIONS"

    def test_display_names_match(self):
        for key, section in CANONICAL_SECTIONS.items():
            assert CANONICAL_SECTION_DISPLAY_NAMES[key] == section.display_name

    def test_default_sections_are_resolvable(self):
        for name in DEFAULT_SECTIONS:
            assert canonicalize_section_name(name) is not None, f"{name!r} not resolvable"
