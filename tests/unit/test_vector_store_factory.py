"""Trace: docs/test-plan.md §5 Vector store factory (CM-7, ADR-0016): OpenSearch is the only backend."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[object]]:
    """Replace the OpenSearch store with a class that records each construction."""
    import app.components.retrieval.opensearch_store as opensearch_module
    import app.components.retrieval.vector_store as module

    instances: list[object] = []

    class FakeOpenSearch:
        def __init__(self) -> None:
            instances.append(self)

    monkeypatch.setattr(opensearch_module, "OpenSearchVectorStore", FakeOpenSearch)
    module.reset_vector_store()
    yield instances
    module.reset_vector_store()


def test_factory_defaults_to_opensearch(monkeypatch: pytest.MonkeyPatch, built: list[object]):
    """test-plan §5: unset VECTOR_BACKEND resolves to OpenSearchVectorStore (default since CM-7)."""
    import app.components.retrieval.vector_store as module

    monkeypatch.delenv("VECTOR_BACKEND", raising=False)

    store = module.get_vector_store()

    assert built == [store]


@pytest.mark.parametrize("value", ["opensearch", "OpenSearch", "  OPENSEARCH  "])
def test_factory_selects_opensearch(
    monkeypatch: pytest.MonkeyPatch, built: list[object], value: str
):
    """test-plan §5: VECTOR_BACKEND=opensearch resolves to OpenSearchVectorStore."""
    import app.components.retrieval.vector_store as module

    monkeypatch.setenv("VECTOR_BACKEND", value)

    store = module.get_vector_store()

    assert built == [store]


@pytest.mark.parametrize(
    "value", ["qdrant", "chroma", "Qdrant", " chroma ", "pinecone-for-some-reason"]
)
def test_factory_rejects_every_other_backend(
    monkeypatch: pytest.MonkeyPatch, built: list[object], value: str
):
    """test-plan §5 Invalid VECTOR_BACKEND: the error names the value and the only accepted
    backend, and no store is built (qdrant and chroma are invalid since CM-7)."""
    import app.components.retrieval.vector_store as module

    monkeypatch.setenv("VECTOR_BACKEND", value)

    with pytest.raises(ValueError) as raised:
        module.get_vector_store()

    message = str(raised.value).lower()
    assert value.strip().lower() in message
    assert "opensearch" in message
    assert built == []
