"""
RAG Embeddings Module
------------------------------

This module provides text embeddings using the ollama's embedding model.

Model used qwen3-embedding:4b model.

Usage:
    from rag.embeddings import get_embeddings, embed_texts, embed_query

    # Embed multiple texts (for indexing)
    texts = ["Risk factor 1...", "Risk factor 2..."]
    vectors = embed_texts(texts)

    # Embed a query (for search)
    query_vector = embed_query("What are the main risks?")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from loguru import logger

from configs.config import settings

# Data Models

@dataclass
class EmbeddingResult:
    """
    Result of embedding operation.

    Attributes:
        vectors: list of embedding vectors
        model: Model used for embedding
        dimensions: Vector dimensions
        total_tokens: Total tokens processed
        error: Error message if embedding failed
    """
    vectors: list[list[float]]
    model: str
    dimensions: int
    total_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return if embedding was successful else error"""
        return self.error is None and len(self.vectors) > 0

    @property
    def count(self) -> int:
        """Number of vectors generated"""
        return len(self.vectors)



# Langchain Embeddings (for ChromaDB)

def get_embeddings(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaEmbeddings:
    """
    Get Langchain-compatible OllamaEmbeddings instance.

    This is the primary way to get embeddings for ChromaDB and other Langchain integrated vector stores.

    Args:
        model: Ollama embedding model name. Defaults to settings.ollama.embed_model
        base_url: Ollama server URL. Defaults to settings.ollama.base_url

    Returns:
        OllamaEmbeddings instance

    Example:
        embeddings = get_embeddings()

        # Embed documents
        vectos = embeddings.embed_documents(["text1", "text2"])

        # Embed query
        query_vector = embeddings.embed_query("search query")
    """
    model = model or settings.ollama.embed_model
    base_url = base_url or settings.ollama.base_url

    logger.info(f"Creating OllamaEmbeddings (model={model})")

    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts synchronously."""
    return get_embeddings().embed_documents(texts)

def embed_query(query: str) -> list[float]:
    """Embed a search query synchronously."""
    return get_embeddings().embed_query(query)

# Async embedding helper

class EmbeddingsClient:
    """
    Async wrapper around OllamaEmbeddings.

    Example:
    ----------------------
        async with EmbeddingsClient() as client:
            # Embed multiple texts/Documents
            results = await client.embed_texts(["text1", "text2"])

            # Embed single query
            vector = await client.embed_query("search query")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Inititalize Embeddings client.

        Args:
            model: Ollama embedding model name
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model or settings.ollama.embed_model
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.timeout = timeout or settings.ollama.timeout

        self._embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )
        self._dimensions: Optional[int] = None

        logger.info(f"EmbeddingsClient initialized (model={self.model})")

    async def __aenter__(self):
        """Async context manage entry."""
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """Async context manager exit."""
        return None

    async def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        result = await self.embed_texts([text])
        if result.success:
            return result.vectors[0]
        raise RuntimeError(f"Embedding failed: {result.error}")

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:

        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call

        Returns:
            EmbeddingResult with vectors
        """
        batch_size = batch_size or settings.ollama.embed_batch_size
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=0,
            )
        logger.info(f"Embedding {len(texts)} texts with {self.model}")

        try:
            all_vectors: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = await asyncio.to_thread(
                    self._embeddings.embed_documents,
                    batch,
                )
                all_vectors.extend(batch_vectors)

            dimensions = len(all_vectors[0]) if all_vectors else 0
            self._dimensions = dimensions

            logger.info(f"Successfully embedded {len(texts)} texts -> {dimensions}D vectors")

            return EmbeddingResult(
                vectors=all_vectors,
                model=self.model,
                dimensions=dimensions,
            )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=0,
                error=str(e),
            )

    async def embed_query(
        self,
        query: str
    ) -> list[float]:
        """
        Embed a search query.

        Args:
            query: Search query

        Returns:
            Embedding vector as a list of float values.
        """
        return await asyncio.to_thread(self._embeddings.embed_query, query)

    @property
    def dimensions(self) -> int:
        """Get vector dimensions

        Returns dimesions from last embedding operation, or default for known models.
        """

        if self._dimensions:
            return self._dimensions

        # known dimensions for common models
        known_dims = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
            "qwen3-embedding:0.6b": 1024,
            "qwen3-embedding:4b": 2560,
            "qwen3-embedding:8b": 4096,
        }

        for model_name, dims in known_dims.items():
            if model_name in self.model:
                return dims
        return 2560

# Wrapper convenience functions

async def embed_texts_async(texts: list[str]) -> EmbeddingResult:
    """
    Embed multiple texts asynchronously. (Wrapper convenience function)

    Args:
        texts: List of texts to embed

    Returns:
        EmbeddingResult with vectors
    """
    async with EmbeddingsClient() as client:
        return await client.embed_texts(texts)

async def embed_query_async(query: str) -> list[float]:
    """
    Embed a search query asynchronously. (Wrapper convenience function)

    Args:
        query: search query

    Returns:
        Query embedding vector
    """
    async with EmbeddingsClient() as client:
        return await client.embed_query(query)
