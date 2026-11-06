"""OpenSearch-specific store behavior (test-plan §5, §21; CM-2)."""

from __future__ import annotations

import time
from typing import Any

import app.components.retrieval.ingestion as ingestion
from tests.filings import make_filing
from tests.rag.helpers import (
    FakeEmbedder,
    all_chunk_ids,
    field_types,
    knn_dimensions,
    raw_document,
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
    expected = f"ollama:{embedder.model}:{embedder.dimension}"
    assert {chunk.metadata.get("embedding_id") for chunk in chunks} == {expected}
