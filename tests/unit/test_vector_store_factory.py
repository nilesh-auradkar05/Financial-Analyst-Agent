from __future__ import annotations

import pytest


def test_vector_store_factory_defaults_to_qdrant(monkeypatch: pytest.MonkeyPatch):
    import app.components.retrieval.qdrant_store as qdrant_module
    import app.components.retrieval.vector_store as module

    class FakeQdrant:
        pass

    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(qdrant_module, "QdrantVectorStore", FakeQdrant)
    module.reset_vector_store()

    store = module.get_vector_store()

    assert isinstance(store, FakeQdrant)


def test_vector_store_factory_selects_qdrant(monkeypatch: pytest.MonkeyPatch):
    import app.components.retrieval.qdrant_store as qdrant_module
    import app.components.retrieval.vector_store as module

    class FakeQdrant:
        pass

    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(qdrant_module, "QdrantVectorStore", FakeQdrant)
    module.reset_vector_store()

    store = module.get_vector_store()

    assert isinstance(store, FakeQdrant)


def test_vector_store_factory_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch):
    import app.components.retrieval.vector_store as module

    monkeypatch.setenv("VECTOR_BACKEND", "pinecone-for-some-reason")
    module.reset_vector_store()

    with pytest.raises(ValueError, match="Unsupported VECTOR_BACKEND"):
        module.get_vector_store()
