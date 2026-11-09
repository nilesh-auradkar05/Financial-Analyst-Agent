"""OpenSearch-specific store behavior (test-plan §5, §21; CM-2)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest

import app.components.retrieval.ingestion as ingestion
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.filings import make_filing
from tests.rag.helpers import (
    FakeEmbedder,
    all_chunk_ids,
    field_types,
    knn_dimensions,
    raw_create_index,
    raw_document,
    raw_index_documents,
    raw_mapping,
)

KEYWORD_FIELDS = ("ticker", "filing_type", "section_key", "filing_date", "accession_number")


def test_healthcheck_reports_unavailable_within_timeout() -> None:
    """test-plan §5: OpenSearch unavailable -> healthcheck() reports unavailable within its
    timeout; no hang or retry storm."""
    from app.components.retrieval.opensearch_store import OpenSearchVectorStore

    store = OpenSearchVectorStore(
        index_name="cm2-test-unavailable",
        url="http://127.0.0.1:1",
        timeout_s=1,
        embeddings=FakeEmbedder(),
    )

    started = time.monotonic()
    healthy = store.healthcheck()
    elapsed = time.monotonic() - started

    assert healthy is False
    assert elapsed < 3


def test_healthcheck_reports_available_against_running_container(make_store) -> None:
    """test-plan §5: OpenSearch unavailable -> healthcheck() is the availability signal
    (positive side: the local container answers)."""
    assert make_store().healthcheck() is True


def test_index_mapping_creation_is_idempotent_with_typed_fields(
    opensearch_url, index_name, make_store, corpus
) -> None:
    """test-plan §5: Index mapping creation -> idempotent; keyword fields for ticker,
    filing_type, section_key, filing_date, accession_number; vector dimension equals the
    configured embedding dimension."""
    first = make_store(index_name)
    second = make_store(index_name)
    status_before, _ = raw_mapping(opensearch_url, index_name)

    first.add_documents(corpus[:3])
    second.add_documents(corpus[3:6])
    first.add_documents(corpus[:3])

    status_after, properties = raw_mapping(opensearch_url, index_name)
    types = field_types(properties)

    assert status_before == 404
    assert status_after == 200
    assert first.count_documents() == 6
    assert second.count_documents() == 6
    assert knn_dimensions(properties) == [FakeEmbedder.dimension]
    for name in KEYWORD_FIELDS:
        assert "keyword" in types.get(name, set()), f"{name} is not mapped as keyword"


def test_chunk_identity_is_preserved_as_document_id(
    opensearch_url, index_name, make_store, corpus
) -> None:
    """test-plan §5: Chunk identity preserved -> the document id stored for a chunk is its
    chunk_id, so gold labels resolve unchanged."""
    store = make_store(index_name)
    store.add_documents(corpus)

    for document in corpus:
        status, stored = raw_document(opensearch_url, index_name, document.id)
        assert status == 200, f"{document.id} not stored under its chunk_id"
        assert stored["_id"] == document.id
    assert sorted(all_chunk_ids(store)) == sorted(document.id for document in corpus)


def test_ingested_chunks_carry_the_embedding_identifier(
    monkeypatch, make_store, embedder
) -> None:
    """test-plan §21: Embedding identifier recorded -> every ingested chunk carries the
    embedding identifier (provider, model, dimension) in its metadata."""
    store = make_store()
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)
    filing: Any = make_filing("AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28")

    result = ingestion.ingest_filing(filing)

    chunks = store.search("all chunks", n_results=1000).chunks
    assert result.success is True
    assert len(chunks) == store.count_documents() > 0
    expected = f"bedrock:{embedder.model}:{embedder.dimension}"
    assert {chunk.metadata.get("embedding_id") for chunk in chunks} == {expected}


# --- CM-3: vector precision, exact distances, delete paging -------------------------------------


@pytest.mark.parametrize("level", ["1x", "32x"])
def test_store_created_index_is_faiss_hnsw_at_the_configured_compression(
    monkeypatch, opensearch_url, index_name, make_store, corpus, level
) -> None:
    """test-plan §5: Vector precision (CM-3) - the created vector field sets compression_level from
    OPENSEARCH_COMPRESSION_LEVEL and names no engine, so the backend resolves faiss; HNSW on
    cosine similarity."""
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", level)

    make_store(index_name).add_documents(corpus[:3])

    _, properties = raw_mapping(opensearch_url, index_name)
    vector = properties["embedding"]
    assert vector["type"] == "knn_vector"
    assert vector["compression_level"] == level
    assert vector["method"]["engine"] == "faiss"
    assert vector["method"]["name"] == "hnsw"
    assert vector["method"]["space_type"] == "cosinesimil"


_INDEX_OPERATIONS: dict[str, Callable[[Any, list[IndexDocument]], object]] = {
    "search": lambda store, corpus: store.search("risk factors"),
    "add_documents": lambda store, corpus: store.add_documents(corpus[3:5]),
}


@pytest.mark.parametrize("operation", sorted(_INDEX_OPERATIONS))
@pytest.mark.parametrize(
    ("index_level", "store_level"), [("1x", "32x"), ("32x", "1x")], ids=["1x-by-32x", "32x-by-1x"]
)
def test_index_with_another_compression_level_is_refused_naming_both_levels(
    monkeypatch, index_name, make_store, corpus, operation, index_level, store_level
) -> None:
    """test-plan §5: Vector precision (CM-3) - an existing index whose compression_level differs from
    the configured one is refused with an IndexCompressionMismatchError naming both levels."""
    from app.components.retrieval.opensearch_store import IndexCompressionMismatchError

    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", index_level)
    make_store(index_name).add_documents(corpus[:3])
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", store_level)
    other = make_store(index_name)

    with pytest.raises(IndexCompressionMismatchError) as refused:
        _INDEX_OPERATIONS[operation](other, corpus)

    assert index_level in str(refused.value)
    assert store_level in str(refused.value)


def test_index_without_compression_level_counts_as_full_precision(
    opensearch_url, index_name, make_store, corpus
) -> None:
    """test-plan §5: Vector precision (CM-3) - an index with no compression_level (absent counts as
    1x) is accepted by a store configured for 1x."""
    raw_create_index(opensearch_url, index_name, FakeEmbedder.dimension)
    raw_index_documents(opensearch_url, index_name, corpus[:3], FakeEmbedder.dimension)

    result = make_store(index_name).search("risk factors")

    assert result.total_results == 3


@pytest.mark.parametrize("operation", sorted(_INDEX_OPERATIONS))
def test_index_without_compression_level_is_refused_at_32x(
    monkeypatch, opensearch_url, index_name, make_store, corpus, operation
) -> None:
    """test-plan §5: Vector precision (CM-3) - an index with no compression_level counts as 1x, so a
    store configured for 32x refuses it, naming both levels."""
    from app.components.retrieval.opensearch_store import IndexCompressionMismatchError

    raw_create_index(opensearch_url, index_name, FakeEmbedder.dimension)
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", "32x")
    store = make_store(index_name)

    with pytest.raises(IndexCompressionMismatchError) as refused:
        _INDEX_OPERATIONS[operation](store, corpus)

    assert "32x" in str(refused.value)
    assert "1x" in str(refused.value)


def test_32x_index_returns_the_same_ids_order_and_distances_as_1x(
    monkeypatch, opensearch_url, new_index, make_store, corpus
) -> None:
    """test-plan §5: Exact distances at any precision (CM-3) - on the same corpus and query a 32x
    index returns the same ids in the same order and the same distance (within 1e-4) as a 1x index."""
    full_index, compressed_index = new_index(), new_index()
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", "1x")
    full = make_store(full_index)
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", "32x")
    compressed = make_store(compressed_index)
    full.add_documents(corpus)
    compressed.add_documents(corpus)

    assert raw_mapping(opensearch_url, full_index)[1]["embedding"]["compression_level"] == "1x"
    assert raw_mapping(opensearch_url, compressed_index)[1]["embedding"]["compression_level"] == "32x"
    queries: list[tuple[str, SearchFilters | None, int]] = [
        ("supply chain disruption competition", None, 5),
        ("interest rate sensitivity hedging", None, 10),
        ("net sales increased gross margin", SearchFilters(ticker="AAPL"), 5),
        ("risk factors", SearchFilters(ticker="MSFT", filing_type="10-K"), 4),
    ]
    for query, filters, n_results in queries:
        expected = full.search(query, filters=filters, n_results=n_results).chunks
        actual = compressed.search(query, filters=filters, n_results=n_results).chunks

        assert expected, f"{query!r} found nothing at 1x"
        assert [chunk.id for chunk in actual] == [chunk.id for chunk in expected], query
        for got, want in zip(actual, expected):
            assert got.distance == pytest.approx(want.distance, abs=1e-4), (query, got.id)


def test_delete_by_ticker_deletes_every_match_past_the_page_size(
    index_name, make_store, corpus
) -> None:
    """test-plan §5: Delete by ticker (CM-3) - more matching ids than the 500-hit page are all
    deleted and counted; other tickers survive."""
    template = corpus[0].metadata
    assert template["ticker"] == "AAPL"
    bulk = [
        IndexDocument(
            id=f"bulk_{number:05d}",
            text=f"bulk passage {number} supply chain",
            metadata={**template, "chunk_id": f"bulk_{number:05d}", "chunk_index": number},
        )
        for number in range(1100)
    ]
    other = [document for document in corpus if document.metadata["ticker"] == "MSFT"]
    store = make_store(index_name)
    store.add_documents(bulk + other)
    assert store.count_documents() == 1100 + len(other)

    deleted = store.delete_by_ticker("AAPL")

    assert deleted == 1100
    assert store.count_documents() == len(other)
