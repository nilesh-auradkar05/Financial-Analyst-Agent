"""Tests for the citation registry and verification in agents.graph.

Item 13: The old approach told the LLM "use [1]" then separately numbered
sources — hoping they matched.  The registry pattern builds a numbered list
BEFORE the LLM call and verifies used indices after.
"""

from agents.graph import (
    _build_citation_registry,
    _extract_used_citations,
    _format_registry_for_prompt,
)

# Easy

class TestBuildRegistry:
    def test_empty_state(self):
        state = {"news_articles": [], "filing_chunks": []}
        registry = _build_citation_registry(state)
        assert registry == []

    def test_news_only(self, sample_news_articles):
        state = {"news_articles": sample_news_articles, "filing_chunks": []}
        registry = _build_citation_registry(state)
        assert len(registry) == 3
        assert registry[0]["index"] == 1
        assert registry[0]["type"] == "news"
        assert registry[2]["index"] == 3

    def test_filings_only(self, sample_filing_chunks):
        state = {"news_articles": [], "filing_chunks": sample_filing_chunks}
        registry = _build_citation_registry(state)
        assert len(registry) == 2
        assert all(r["type"] == "sec_filing" for r in registry)

    def test_mixed_sources_numbered_sequentially(
        self, sample_news_articles, sample_filing_chunks,
    ):
        state = {
            "news_articles": sample_news_articles,
            "filing_chunks": sample_filing_chunks,
        }
        registry = _build_citation_registry(state)
        indices = [r["index"] for r in registry]
        assert indices == list(range(1, len(registry) + 1))


# Medium

class TestFormatRegistryForPrompt:
    def test_empty_registry(self):
        assert _format_registry_for_prompt([]) == ""

    def test_contains_source_labels(self, sample_news_articles):
        state = {"news_articles": sample_news_articles, "filing_chunks": []}
        registry = _build_citation_registry(state)
        text = _format_registry_for_prompt(registry)
        assert "[1]" in text
        assert "(news)" in text
        assert "Available Sources" in text


# Hard
class TestExtractUsedCitations:
    def test_no_citations(self):
        assert _extract_used_citations("No citations here.") == set()

    def test_single_citation(self):
        assert _extract_used_citations("Revenue grew [1].") == {1}

    def test_multiple_citations(self):
        memo = "Strong growth [1]. Risk factors [3]. See also [1][2]."
        assert _extract_used_citations(memo) == {1, 2, 3}

    def test_ignores_non_numeric_brackets(self):
        assert _extract_used_citations("Use [N] notation.") == set()

    def test_adjacent_citations(self):
        assert _extract_used_citations("See [1][2][3].") == {1, 2, 3}


class TestOrphanDetection:
    """Verify that we can detect citations the LLM invented."""

    def test_all_valid(self, sample_news_articles):
        state = {"news_articles": sample_news_articles, "filing_chunks": []}
        registry = _build_citation_registry(state)
        valid = {r["index"] for r in registry}
        used = _extract_used_citations("See [1] and [2].")
        orphans = used - valid
        assert orphans == set()

    def test_orphan_detected(self):
        registry = [{"index": 1, "type": "news", "title": "A"}]
        valid = {r["index"] for r in registry}
        used = _extract_used_citations("See [1] and [99].")
        orphans = used - valid
        assert 99 in orphans

    def test_no_false_positives_on_empty_memo(self):
        used = _extract_used_citations("")
        assert used == set()
