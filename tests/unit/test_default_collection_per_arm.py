"""Trace: docs/test-plan.md §21 "One index per arm" (CM-6e): every vector store names its default
collection from the embedding identifier, the default arm included, so a 1024-d Foundry Cohere
arm never writes into a collection holding another arm's vectors.

The Qdrant and Chroma stores are built as the factory builds them. The Qdrant client is replaced
at the system boundary by a client that never connects; embeddings are a fixed fake, because only
the collection name is under test. Settings come from environment variables only.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app.components.retrieval.qdrant_store as qdrant_module
import app.components.retrieval.vector_store as vector_store_module
from app.config import settings
from tests.embedding_arms import (
    BEDROCK_COHERE,
    FOUNDRY_COHERE,
    Arm,
    embeddings_api,
    use_embedding_settings,
)

BASE = "sec_filings"
DEFAULT_NAME = "sec_filings__foundry-cohere-cohere-embed-v5-fast-1024"


class _NoConnectionClient:
    """Stands in for QdrantClient: reports an existing collection and holds nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def get_collection(self, **kwargs: Any) -> object:
        return object()

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected call to the vector database: {name}")


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


Build = Callable[[], Any]
BACKENDS = ["qdrant", "chroma"]


@pytest.fixture
def build_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[str], Build]:
    """`build_store(backend)` returns a function that builds that backend's store through the factory."""
    monkeypatch.setattr(qdrant_module, "QdrantClient", _NoConnectionClient)
    monkeypatch.setattr(qdrant_module, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(settings.chroma, "persist_dir", str(tmp_path / "chroma"))

    def for_backend(backend: str) -> Build:
        def build() -> Any:
            monkeypatch.setenv("VECTOR_BACKEND", backend)
            vector_store_module.reset_vector_store()
            return vector_store_module.get_vector_store()

        return build

    return for_backend


def _arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: Arm, **extra: str) -> None:
    use_embedding_settings(monkeypatch, tmp_path, {**arm.env, **extra})
    monkeypatch.setattr(settings.chroma, "persist_dir", str(tmp_path / "chroma"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_default_collection_is_named_from_the_default_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_store: Callable[[str], Build],
    backend: str,
) -> None:
    """test-plan §21: One index per arm (the default arm no longer maps to `sec_filings`)."""
    _arm(monkeypatch, tmp_path, FOUNDRY_COHERE)

    store = build_store(backend)()

    assert store.collection_name == embeddings_api.embedding_index_name(BASE) == DEFAULT_NAME
    assert store.collection_name != BASE


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_arms_get_different_default_collections_on_the_same_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_store: Callable[[str], Build],
    backend: str,
) -> None:
    """test-plan §21: One index per arm (distinct and stable per arm)."""
    build = build_store(backend)
    _arm(monkeypatch, tmp_path, FOUNDRY_COHERE)
    first = build().collection_name
    _arm(monkeypatch, tmp_path, BEDROCK_COHERE)
    other = build().collection_name
    _arm(monkeypatch, tmp_path, FOUNDRY_COHERE)
    again = build().collection_name

    assert first == again == DEFAULT_NAME
    assert other == "sec_filings__bedrock-cohere-embed-v4-0-1024"


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_explicit_collection_name_wins_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_store: Callable[[str], Build],
    backend: str,
) -> None:
    """test-plan §21: One index per arm (explicit names are not rewritten)."""
    _arm(monkeypatch, tmp_path, FOUNDRY_COHERE)
    build_store(backend)  # patches the boundary
    name = f"explicit_{uuid.uuid4().hex[:8]}"
    store: Any

    if backend == "qdrant":
        store = qdrant_module.QdrantVectorStore(collection_name=name, url="http://qdrant-test:6333")
    else:
        store = vector_store_module.ChromaDBVectorStore(
            collection_name=name, persist_directory=str(tmp_path / "explicit-chroma")
        )

    assert store.collection_name == name


def test_the_qdrant_collection_variable_wins_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_store: Callable[[str], Build],
) -> None:
    """test-plan §21: One index per arm (QDRANT_COLLECTION_NAME is an explicit name)."""
    _arm(monkeypatch, tmp_path, FOUNDRY_COHERE, QDRANT_COLLECTION_NAME="from_the_environment")

    store = build_store("qdrant")()

    assert store.collection_name == "from_the_environment"
