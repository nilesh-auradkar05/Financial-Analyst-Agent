"""
RAG Embeddings Module
------------------------------

This module provides text embeddings using the ollama's embedding model.

Model used nomic-embed-text model.

Usage:
    from rag.embeddings import get_embeddings, embed_texts, embed_query

    # Embed multiple texts (for indexing)
    texts = ["Risk factor 1...", "Risk factor 2..."]
    vectors = embed_texts(texts)

    # Embed a query (for search)
    query_vector = embed_query("What are the main risks?")
"""

import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from loguru import logger

from langchain_ollama import OllamaEmbeddings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Direct Ollama API client

class EmbeddingsClient:
    """
    Direct Client for Ollama's embeddings API.

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

        self._client: Optional[httpx.AsyncClient] = None
        self._dimensions: Optional[int] = None
        
        logger.info(f"EmbeddingsClient initialized (model={self.model})")

    async def __aenter__(self):
        """Async context manage entry"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HHTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with EmbeddingsClient() as client:")
        return self._client

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
        else:
            raise RuntimeError(f"Embedding failed: {result.error}")

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> EmbeddingResult:

        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call

        Returns:
            EmbeddingResult with vectors
        """
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=0,
            )
        logger.info(f"Embedding {len(texts)} texts with {self.model}")

        all_vectors = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_vectors = await self._embed_batch(batch)
                all_vectors.extend(batch_vectors)

            # Get dimensions from first vector
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

    async def _embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        """
        Embed a batch of texts with Ollama API.

        Args:
            texts: Batch of texts

        Returns:
            List of embedding vectors
        """

        vectors = []

        for text in texts:
            # Ollama's API expects one text a time
            payload = {
                "model": self.model,
                "input": text,
            }

            response = await self.client.post("/api/embed", json=payload)
            response.raise_for_status()

            data = response.json()

            # Response format: {"embedding": [[vector]]}
            embedding = data.get("embeddings", [[]])[0]
            vectors.append(embedding)

        return vectors
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
        return await self.embed_single(query)

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
        }

        for model_name, dims in known_dims.items():
            if model_name in self.model:
                return dims
        return 768

# Wrapper convenience functions

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts synchronously. (Wrapper convenience function)

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Example:
        vectors = embed_texts(["Apple's revenue grew", "Stock declined"])
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)

def embed_query(query: str) -> list[float]:
    """
    Embed a search query synchronously. (Wrapper convenience function)

    Args:
        query: Search query

    Returns:
        Query embedding vector
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(query)

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

# Testing

async def _main():
    """Test the embeddings module."""
    import sys

    print(f"\n{'='*60}")
    print("Embeddings Module - Testing")
    print(f"{'='*60}\n")

    # Test Langchain embeddings
    print("1. Langchain Embeddings....")
    embeddings = get_embeddings()

    test_texts = [
        "Apple reported strong earning growth.",
        "The company faces regulatory challenges.",
        "Stock prices surged after the announcement.",
    ]

    try:
        vectors = embeddings.embed_documents(test_texts)
        print(f"    Embedded {len(vectors)} texts")
        print(f"    Dimensions: {len(vectors[0])}")
        print(f"    Sample vector (first 5 dims): {vectors[0][:5]}\n")
    except Exception as e:
        print(f"    Failed: {e}\n")
        return

    # Test query embedding
    print("2. Query Embedding....")
    query = "What are the risks?"
    query_vector = embeddings.embed_query(query)
    print(f"    Query: {query}\n")
    print(f"    Dimensions: {len(query_vector)}\n")

    # Test Async embeddings
    print("3. Async Embeddings....")
    async with EmbeddingsClient() as client:
        result = await client.embed_texts(test_texts)

        if result.success:
            print(f"    Async Embedded {result.count} texts")
            print(f"    Dimensions: {result.dimensions}\n")
        else:
            print(f"    Async embedding failed: {result.error}\n")

    # Test Similarity
    print("4. Similarity Test....")
    import math

    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    # Compare query to each text
    for i, (text, vec) in enumerate(zip(test_texts, vectors)):
        sim = cosine_similarity(query_vector, vec)
        print(f"    Query vs Text {i+1}: {sim:.4f}")
        print(f"    Text: {text[:50]}...")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    asyncio.run(_main())