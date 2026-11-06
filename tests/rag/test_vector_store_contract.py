"""RetrievalStore contract against two separately populated OpenSearch indexes
(test-plan §4; "both backends" means two indexes holding the same corpus)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

import app.components.retrieval.vector_store as vector_store_module
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.rag.helpers import (
    REQUIRED_METADATA_KEYS,
    all_chunk_ids,
    build_corpus,
    find_backend_objects,
    public_attribute_values,
)

CORPUS = build_corpus()
CORPUS_BY_ID = {document.id: document for document in CORPUS}
MD_AND_A_CHUNK_ID = "0000320193-25-000001_md&a_000"

FILTER_CASES = [
    pytest.param(SearchFilters(ticker="aapl"), {"ticker": "AAPL"}, id="ticker-case-insensitive"),
    pytest.param(
        SearchFilters(ticker="MSFT", section_key="risk_factors"),
        {"ticker": "MSFT", "section_key": "risk_factors"},
        id="ticker-and-section-key",
    ),
    pytest.param(SearchFilters(filing_type="10-Q"), {"filing_type": "10-Q"}, id="filing-type"),
    pytest.param(
        SearchFilters(filing_date="2025-06-30"), {"filing_date": "2025-06-30"}, id="filing-date"
    ),
    pytest.param(
        SearchFilters(extra={"accession_number": "0000320193-25-000001"}),
        {"accession_number": "0000320193-25-000001"},
        id="accession-number",
    ),
    pytest.param(
        SearchFilters(extra={"section": "Item 1A. Risk Factors"}),
        {"section": "Item 1A. Risk Factors"},
        id="section-display-name",
    ),
    pytest.param(
        SearchFilters(extra={"chunk_id": MD_AND_A_CHUNK_ID}),
        {"chunk_id": MD_AND_A_CHUNK_ID},
        id="chunk-id-with-ampersand",
    ),
]


def matching_ids(expected_metadata: dict[str, Any]) -> set[str]:
    return {
        document.id
        for document in CORPUS
        if all(document.metadata.get(key) == value for key, value in expected_metadata.items())
    }


@pytest.fixture
def stores(make_store) -> tuple[Any, Any]:
    """Two separately populated indexes holding the same corpus."""
    first, second = make_store(), make_store()
    first.add_documents(CORPUS)
    second.add_documents(CORPUS)
    return first, second


def test_same_corpus_gives_identical_total_count(stores) -> None:
    """test-plan §4: Same corpus, both backends -> identical total count_documents()."""
    first, second = stores

    assert first.count_documents() == second.count_documents() == len(CORPUS)
    assert first.get_stats()["total_documents"] == len(CORPUS)
    assert first.get_stats()["backend"] == "opensearch"


@pytest.mark.parametrize(("filters", "expected_metadata"), FILTER_CASES)
def test_same_filter_gives_identical_filtered_count(stores, filters, expected_metadata) -> None:
    """test-plan §4: Same filter -> identical filtered count."""
    first, second = stores
    expected = len(matching_ids(expected_metadata))

    assert expected > 0
    assert first.count_documents(filters) == second.count_documents(filters) == expected


@pytest.mark.parametrize(("filters", "expected_metadata"), FILTER_CASES)
def test_filtered_search_returns_only_matching_metadata(
    stores, filters, expected_metadata
) -> None:
    """test-plan §5: Filtered count/search -> only matching metadata returned."""
    first, _ = stores

    result = first.search("risk disclosures", filters=filters, n_results=50)

    assert {chunk.id for chunk in result.chunks} == matching_ids(expected_metadata)
    for chunk in result.chunks:
        for key, value in expected_metadata.items():
            assert chunk.metadata[key] == value


def test_ticker_and_section_helpers_return_only_matching_metadata(stores) -> None:
    """test-plan §5: Filtered count/search -> only matching metadata returned (ticker and
    section-key convenience searches)."""
    first, _ = stores

    by_ticker = first.search_by_ticker("supply chain", "msft", n_results=50)
    by_sections = first.search_sections(
        "MSFT", ["business", "market_risk"], n_results=50, query="exposure"
    )

    assert {chunk.id for chunk in by_ticker.chunks} == matching_ids({"ticker": "MSFT"})
    expected_sections = {
        document.id
        for document in CORPUS
        if document.metadata["ticker"] == "MSFT"
        and document.metadata["section_key"] in {"business", "market_risk"}
    }
    assert {chunk.id for chunk in by_sections.chunks} == expected_sections


def test_same_query_and_filter_give_metadata_valid_evidence_packets(stores) -> None:
    """test-plan §4: Same query + filter -> metadata-valid EvidencePackets on both backends."""
    target = CORPUS_BY_ID["0000789019-25-000001_risk_factors_001"]
    filters = SearchFilters(ticker="MSFT", filing_type="10-K")

    for store in stores:
        result = store.search(target.text, filters=filters, n_results=10)
        packets = result.to_evidence_packets()

        assert [packet.rank for packet in packets] == list(range(1, len(packets) + 1))
        assert packets[0].chunk_id == target.id
        assert packets[0].retrieval_score == pytest.approx(1.0, abs=1e-3)
        scores = [packet.retrieval_score for packet in packets]
        assert scores == sorted(scores, reverse=True)
        assert {packet.chunk_id for packet in packets} == matching_ids(
            {"ticker": "MSFT", "filing_type": "10-K"}
        )
        for packet in packets:
            stored = CORPUS_BY_ID[packet.chunk_id]
            assert packet.ticker == stored.metadata["ticker"]
            assert packet.filing_type == stored.metadata["filing_type"]
            assert packet.filing_date == stored.metadata["filing_date"]
            assert packet.accession_number == stored.metadata["accession_number"]
            assert packet.section_name == stored.metadata["section_name"]
            assert packet.source_url == stored.metadata["source_url"]
            assert packet.text == stored.text
            assert "text" not in packet.metadata
        assert result.chunks[0].distance == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param(SearchFilters(ticker="ZZZZ"), id="unknown-ticker"),
        pytest.param(
            SearchFilters(ticker="AAPL", filing_type="10-Q", section_key="business"),
            id="no-chunk-satisfies-all-clauses",
        ),
    ],
)
def test_no_matches_returns_empty_without_raising(stores, filters) -> None:
    """test-plan §4: No matches -> returns empty list, never raises."""
    first, _ = stores

    result = first.search("anything at all", filters=filters)

    assert result.chunks == []
    assert result.total_results == 0
    assert result.has_results is False
    assert first.count_documents(filters) == 0


def test_missing_index_reads_as_empty_without_raising(make_store) -> None:
    """test-plan §4: No matches -> returns empty list, never raises (index not yet created)."""
    store = make_store()

    assert store.search("anything at all").chunks == []
    assert store.count_documents() == 0
    assert store.delete_by_ticker("AAPL") == 0


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param(SearchFilters(extra={"not_a_metadata_field": "x"}), id="unknown-key"),
        pytest.param(SearchFilters(extra={"ticker": ["AAPL", "MSFT"]}), id="list-value"),
        pytest.param(SearchFilters(extra={"section_key": {"eq": "business"}}), id="dict-value"),
    ],
)
@pytest.mark.parametrize("operation", ["search", "count_documents"])
def test_invalid_filter_is_a_validation_error(stores, filters, operation) -> None:
    """test-plan §4: Invalid filter -> validation error."""
    first, _ = stores
    call = {
        "search": lambda: first.search("anything", filters=filters),
        "count_documents": lambda: first.count_documents(filters),
    }[operation]

    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("missing_key", REQUIRED_METADATA_KEYS)
def test_upsert_with_missing_metadata_is_rejected_and_writes_nothing(
    make_store, missing_key
) -> None:
    """test-plan §4: Missing-metadata upsert -> rejected (nothing is written)."""
    store = make_store()
    valid = CORPUS[0]
    incomplete = IndexDocument(
        id="incomplete-chunk",
        text="a chunk without complete filing metadata",
        metadata={k: v for k, v in CORPUS[1].metadata.items() if k != missing_key},
    )

    with pytest.raises(ValueError):
        store.add_documents([valid, incomplete])

    assert store.count_documents() == 0


def test_reupserting_same_chunks_never_duplicates_chunk_ids(make_store) -> None:
    """test-plan §4: Re-upsert same chunks -> no duplicate chunk IDs (and the same id is
    overwritten, not appended)."""
    store = make_store()
    edited_target = CORPUS[3]
    edited = replace(edited_target, text="a revised passage about restated segment results")

    store.add_documents(CORPUS)
    store.add_documents(CORPUS)
    store.add_documents([edited])

    ids = all_chunk_ids(store)
    assert store.count_documents() == len(CORPUS)
    assert len(ids) == len(set(ids)) == len(CORPUS)
    top = store.search(edited.text, n_results=1).chunks[0]
    assert top.id == edited_target.id
    assert top.text == edited.text


def test_no_backend_client_object_is_observable_above_the_store(make_store, monkeypatch) -> None:
    """test-plan §4: Backend client leakage -> no backend client object observable above
    RetrievalService (owner reading: nothing from get_vector_store(), any RetrievalStore
    method, or a public attribute of the store is an opensearchpy object)."""
    store = make_store()
    returned: dict[str, Any] = {
        "add_documents": store.add_documents(CORPUS),
        "search": store.search("supply chain", n_results=5),
        "search_sections": store.search_sections("MSFT", ["business"], query="products"),
        "search_by_ticker": store.search_by_ticker("risk", "AAPL"),
        "count_documents": store.count_documents(),
        "get_stats": store.get_stats(),
        "count": store.count,
        "delete_by_ticker": store.delete_by_ticker("AAPL"),
    }
    monkeypatch.setenv("VECTOR_BACKEND", "opensearch")
    vector_store_module.reset_vector_store()
    try:
        configured = vector_store_module.get_vector_store()
    finally:
        vector_store_module.reset_vector_store()

    leaks: list[str] = []
    for name, value in returned.items():
        leaks.extend(find_backend_objects(value, name))
    for label, owner in (("store", store), ("get_vector_store()", configured)):
        assert not hasattr(owner, "client")
        for name, value in public_attribute_values(owner).items():
            leaks.extend(find_backend_objects(value, f"{label}.{name}"))

    assert leaks == []

