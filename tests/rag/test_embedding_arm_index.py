"""Per-arm OpenSearch behaviour (test-plan §21; CM-5a): dimension guard, embedding identifier on
every chunk, and the chunk iterator the arm build copies from.

Runs against the local OpenSearch test container. Embeddings are injected as hashed fakes of the
dimension under test; settings name the arm through environment variables only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import app.components.retrieval.ingestion as ingestion
from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument
from tests.embedding_arms import (
    BEDROCK_COHERE,
    BEDROCK_TITAN,
    FOUNDRY_COHERE,
    Arm,
    embeddings_api,
    use_embedding_settings,
)
from tests.filings import make_filing
from tests.rag.helpers import (
    SizedFakeEmbedder,
    knn_dimensions,
    raw_count,
    raw_create_index,
    raw_index_documents,
    raw_mapping,
)

NO_ENDPOINT = "http://127.0.0.1:1"
EXISTING_DIMENSION = 2560  # an index built with vectors of another size than the configured one
CONFIGURED_DIMENSION = BEDROCK_TITAN.dimension


def _store(url: str, index: str, dimension: int) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(
        index_name=index, url=url, timeout_s=10, embeddings=SizedFakeEmbedder(dimension)
    )


@pytest.mark.parametrize("operation", ["add_documents", "search"])
def test_collection_with_another_vector_size_is_refused_before_any_write_or_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    index_name: str,
    corpus: list[IndexDocument],
    operation: str,
) -> None:
    """test-plan §21: Dimension mismatch refused."""
    use_embedding_settings(monkeypatch, tmp_path, BEDROCK_TITAN.routed_to(NO_ENDPOINT))
    raw_create_index(opensearch_url, index_name, EXISTING_DIMENSION)
    raw_index_documents(opensearch_url, index_name, corpus[:3], EXISTING_DIMENSION)
    store = _store(opensearch_url, index_name, CONFIGURED_DIMENSION)
    actions = {
        "add_documents": lambda: store.add_documents(corpus[3:]),
        "search": lambda: store.search("supply chain risk"),
    }

    with pytest.raises(embeddings_api.EmbeddingDimensionMismatchError) as refused:
        actions[operation]()

    assert issubclass(embeddings_api.EmbeddingDimensionMismatchError, ValueError)
    assert str(EXISTING_DIMENSION) in str(refused.value)
    assert str(CONFIGURED_DIMENSION) in str(refused.value)
    assert raw_count(opensearch_url, index_name) == 3


@pytest.mark.parametrize("already_exists", [True, False], ids=["existing-index", "new-index"])
def test_collection_with_the_configured_vector_size_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    index_name: str,
    corpus: list[IndexDocument],
    already_exists: bool,
) -> None:
    """test-plan §21: Dimension mismatch refused (a matching or new collection is not refused and
    is created with the configured dimension)."""
    use_embedding_settings(monkeypatch, tmp_path, BEDROCK_TITAN.routed_to(NO_ENDPOINT))
    if already_exists:
        raw_create_index(opensearch_url, index_name, CONFIGURED_DIMENSION)
    store = _store(opensearch_url, index_name, CONFIGURED_DIMENSION)

    store.add_documents(corpus)

    _, properties = raw_mapping(opensearch_url, index_name)
    assert knn_dimensions(properties) == [CONFIGURED_DIMENSION]
    assert store.count_documents() == len(corpus)
    assert len(store.search("supply chain risk", n_results=3).chunks) == 3


@pytest.mark.parametrize("arm", [FOUNDRY_COHERE, BEDROCK_COHERE], ids=lambda arm: arm.name)
def test_every_ingested_chunk_carries_the_embedding_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    index_name: str,
    arm: Arm,
) -> None:
    """test-plan §21: Embedding identifier recorded."""
    use_embedding_settings(monkeypatch, tmp_path, arm.routed_to(NO_ENDPOINT))
    store = _store(opensearch_url, index_name, arm.dimension)
    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)
    filing: Any = make_filing(
        "AAPL", accession_number="0000320193-25-000001", filing_date="2025-09-28"
    )

    result = ingestion.ingest_filing(filing)

    chunks = store.search("all chunks", n_results=1000).chunks
    assert result.success is True
    assert len(chunks) == store.count_documents() > 0
    assert embeddings_api.embedding_identifier() == arm.identifier
    assert {chunk.metadata.get("embedding_id") for chunk in chunks} == {arm.identifier}


def test_iter_documents_yields_every_chunk_with_text_and_metadata_but_no_vector(
    make_store: Any, corpus: list[IndexDocument]
) -> None:
    """test-plan §21: Arm built from the same chunks (the chunk iterator the build copies from)."""
    store = make_store()
    store.add_documents(corpus)

    documents = list(store.iter_documents())

    by_id = {document.id: document for document in documents}
    assert len(documents) == len(by_id) == len(corpus)
    for original in corpus:
        copy = by_id[original.id]
        assert copy.text == original.text
        assert "embedding_id" in copy.metadata
        assert {k: v for k, v in copy.metadata.items() if k != "embedding_id"} == original.metadata


def test_iter_documents_pages_through_an_index_larger_than_one_page(
    make_store: Any, corpus: list[IndexDocument]
) -> None:
    """test-plan §21: Arm built from the same chunks (every chunk of a large baseline is visited)."""
    copies = 120
    many = [
        IndexDocument(
            id=f"{document.id}-{n}",
            text=f"{document.text} copy {n}",
            metadata={**document.metadata, "chunk_id": f"{document.id}-{n}"},
        )
        for n in range(copies)
        for document in corpus
    ]
    store = make_store()
    store.add_documents(many)

    ids = [document.id for document in store.iter_documents()]

    assert len(many) == copies * len(corpus) > 1000
    assert len(ids) == len(set(ids)) == len(many)
