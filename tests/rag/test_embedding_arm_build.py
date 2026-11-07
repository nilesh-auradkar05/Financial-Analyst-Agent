"""Building an embedding arm's index from the baseline chunks (test-plan §21; CM-5a, layer I).

The baseline is a 2560-d index stamped with the default identifier; the arm is a 1024-d Bedrock
Cohere index. Both live on the local OpenSearch test container, each filled through the store with a
hashed fake embedder of its dimension. The build runs under the arm's settings, as the CLI does, so
reading the baseline must not be refused because its vector size differs from the configured one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.embedding_arms import BEDROCK_COHERE, OLLAMA, use_embedding_settings
from tests.rag.helpers import SizedFakeEmbedder, raw_vector

NO_ENDPOINT = "http://127.0.0.1:1"
SOURCE_EMBEDDER = SizedFakeEmbedder(OLLAMA.dimension, model="qwen-like")
TARGET_EMBEDDER = SizedFakeEmbedder(BEDROCK_COHERE.dimension, model="cohere-like")


def _store(url: str, index: str, embedder: SizedFakeEmbedder) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(index_name=index, url=url, timeout_s=10, embeddings=embedder)


@pytest.fixture
def baseline_and_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    new_index: Callable[[], str],
    corpus: list[IndexDocument],
) -> tuple[Any, Any]:
    """A filled baseline index and an empty arm index, with settings on the arm."""
    use_embedding_settings(monkeypatch, tmp_path, OLLAMA.routed_to(NO_ENDPOINT))
    source = _store(opensearch_url, new_index(), SOURCE_EMBEDDER)
    source.add_documents(corpus)
    use_embedding_settings(monkeypatch, tmp_path, BEDROCK_COHERE.routed_to(NO_ENDPOINT))
    return source, _store(opensearch_url, new_index(), TARGET_EMBEDDER)


def test_arm_is_built_from_the_same_chunks_re_embedding_only_the_vector(
    opensearch_url: str,
    baseline_and_arm: tuple[Any, Any],
    corpus: list[IndexDocument],
) -> None:
    """test-plan §21: Arm built from the same chunks."""
    from evaluation.build_embedding_arm import build_embedding_arm

    source, target = baseline_and_arm
    per_ticker = Counter(document.metadata["ticker"] for document in corpus)

    counts = build_embedding_arm(source, target)

    assert counts == dict(per_ticker)
    assert source.count_documents() == len(corpus)
    for ticker, expected in per_ticker.items():
        filters = SearchFilters(ticker=ticker)
        assert source.count_documents(filters) == target.count_documents(filters) == expected
    copied = {document.id: document for document in target.iter_documents()}
    original = {document.id: document for document in source.iter_documents()}
    assert copied.keys() == original.keys() == {document.id for document in corpus}
    for chunk_id, document in copied.items():
        assert document.text == original[chunk_id].text
        assert original[chunk_id].metadata["embedding_id"] == OLLAMA.identifier
        assert document.metadata["embedding_id"] == BEDROCK_COHERE.identifier
        unchanged = {k: v for k, v in document.metadata.items() if k != "embedding_id"}
        assert unchanged == {
            k: v for k, v in original[chunk_id].metadata.items() if k != "embedding_id"
        }
        vector = raw_vector(opensearch_url, target.index_name, chunk_id)
        assert len(vector) == BEDROCK_COHERE.dimension
        assert vector == pytest.approx(TARGET_EMBEDDER.embed_documents([document.text])[0])


def test_build_fails_when_the_arm_ends_with_different_chunk_counts_than_the_baseline(
    baseline_and_arm: tuple[Any, Any],
    corpus: list[IndexDocument],
) -> None:
    """test-plan §21: Arm built from the same chunks (counts must match per ticker)."""
    from evaluation.build_embedding_arm import build_embedding_arm

    source, target = baseline_and_arm
    stray = IndexDocument(
        id="stray-chunk-not-in-the-baseline",
        text="a leftover chunk from an earlier build",
        metadata={**corpus[0].metadata, "chunk_id": "stray-chunk-not-in-the-baseline"},
    )
    target.add_documents([stray])

    with pytest.raises(Exception) as failed:  # noqa: B017 - the failure type is not part of the contract
        build_embedding_arm(source, target)

    assert not isinstance(failed.value, ImportError | AttributeError | TypeError)
