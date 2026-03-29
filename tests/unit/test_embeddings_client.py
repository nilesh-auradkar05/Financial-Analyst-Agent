"""Tests for rag.embeddings — replaces the old ``_main()`` smoke test.

These tests mock the Ollama server so they run in CI without GPU.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag.embeddings import (
    EmbeddingResult,
    EmbeddingsClient,
    embed_query,
    embed_query_async,
    embed_texts,
    embed_texts_async,
    get_embeddings,
)

# Easy: module structure

class TestGetEmbeddings:
    def test_returns_ollama_embeddings(self):
        emb = get_embeddings()
        assert hasattr(emb, "embed_documents")
        assert hasattr(emb, "embed_query")

    def test_custom_model(self):
        emb = get_embeddings(model="nomic-embed-text")
        assert emb.model == "nomic-embed-text"


# Medium: sync wrappers

class TestSyncWrappers:
    @patch("rag.embeddings.get_embeddings")
    def test_embed_texts(self, mock_get):
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get.return_value = mock_emb

        result = embed_texts(["text1", "text2"])
        assert len(result) == 2
        mock_emb.embed_documents.assert_called_once()

    @patch("rag.embeddings.get_embeddings")
    def test_embed_query(self, mock_get):
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.5, 0.6]
        mock_get.return_value = mock_emb

        result = embed_query("search term")
        assert len(result) == 2


# Hard: async client

class TestEmbeddingsClient:
    def test_dimensions_known_model(self):
        client = EmbeddingsClient(model="qwen3-embedding:4b")
        assert client.dimensions == 2560

    def test_dimensions_unknown_model_default(self):
        client = EmbeddingsClient(model="custom-model")
        assert client.dimensions == 2560

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self):
        async with EmbeddingsClient() as client:
            result = await client.embed_texts([])
            assert result.success is False  # no vectors
            assert result.count == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with EmbeddingsClient() as client:
            assert client is not None
            assert client.model is not None


# Edge: EmbeddingResult

class TestEmbeddingResult:
    def test_success_with_vectors(self):
        r = EmbeddingResult(vectors=[[0.1]], model="test", dimensions=1)
        assert r.success is True
        assert r.count == 1

    def test_failure_with_error(self):
        r = EmbeddingResult(vectors=[], model="test", dimensions=0, error="boom")
        assert r.success is False

    def test_empty_no_error_not_success(self):
        r = EmbeddingResult(vectors=[], model="test", dimensions=0)
        assert r.success is False
