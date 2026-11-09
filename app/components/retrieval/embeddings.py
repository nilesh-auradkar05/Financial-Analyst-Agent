"""
RAG Embeddings Module
------------------------------

This module provides text embeddings for the configured arm (settings.embedding):
Cohere Embed v5 fast on Microsoft Foundry (default), Bedrock or Azure OpenAI.

Usage:
    from app.components.retrieval.embeddings import get_embeddings, embed_texts, embed_query

    # Embed multiple texts (for indexing)
    texts = ["Risk factor 1...", "Risk factor 2..."]
    vectors = embed_texts(texts)

    # Embed a query (for search)
    query_vector = embed_query("What are the main risks?")
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from langchain_aws.utils import create_aws_client
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.config import settings

# Cohere Embed v5 fast on Foundry allows 20 requests per minute; stay under it.
FOUNDRY_COHERE_MIN_INTERVAL_SECONDS = 60 / 18
_FOUNDRY_COHERE_MAX_TEXTS = 96
_FOUNDRY_COHERE_MAX_RETRY_WAIT_SECONDS = 60.0

# ponytail: per-process limiter; several processes sharing one deployment need a shared
# limiter (S7.9 cache store).
_pacing_lock = threading.Lock()
_last_request_start: float | None = None


class EmbeddingUnavailableError(RuntimeError):
    """The embedding provider is not configured, unreachable, throttled or returned a bad reply."""


class EmbeddingDimensionMismatchError(ValueError):
    """An index holds vectors of a size other than the configured embedding dimension."""


def embedding_identifier() -> str:
    """Stable name of the configured arm: provider:model:dimensions."""
    arm = settings.embedding
    return f"{arm.provider}:{arm.model}:{arm.dimensions}"


def embedding_index_name(base: str = "sec_filings") -> str:
    """Index name for the configured arm: `base` plus a slug of its identifier."""
    return f"{base}__{re.sub(r'[^a-z0-9]+', '-', embedding_identifier().lower())}"


def _wait_for_request_slot() -> None:
    """Block until a request may start, then claim the slot; an idle process is not delayed."""
    global _last_request_start
    with _pacing_lock:
        if _last_request_start is not None:
            wait = _last_request_start + FOUNDRY_COHERE_MIN_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        _last_request_start = time.monotonic()


def _retry_wait_seconds(retry_after: str | None) -> float:
    try:
        seconds = float(retry_after or "")
    except ValueError:
        return FOUNDRY_COHERE_MIN_INTERVAL_SECONDS
    if not seconds >= 0:
        return FOUNDRY_COHERE_MIN_INTERVAL_SECONDS
    return min(seconds, _FOUNDRY_COHERE_MAX_RETRY_WAIT_SECONDS)


class FoundryCohereEmbeddings(Embeddings):
    """Cohere Embed v5 fast on Microsoft Foundry (ADR-0011); every request is paced process-wide."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _FOUNDRY_COHERE_MAX_TEXTS):
            batch = texts[start : start + _FOUNDRY_COHERE_MAX_TEXTS]
            vectors.extend(self._embed(batch, "search_document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "search_query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        llm = settings.llm
        arm = settings.embedding
        endpoint = (llm.azure_openai_endpoint or "").strip().rstrip("/")
        key = llm.azure_api_key.get_secret_value().strip() if llm.azure_api_key else ""
        missing = [
            name
            for name, value in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_API_KEY", key))
            if not value
        ]
        if missing:
            raise EmbeddingUnavailableError(f"Foundry Cohere embeddings need {' and '.join(missing)}.")
        payload = {
            "model": arm.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
            "output_dimension": arm.dimensions,
        }
        reply = self._post(f"{endpoint}/providers/cohere/v2/embed", key, payload)
        return self._parse_vectors(reply, len(texts), arm.dimensions)

    @staticmethod
    def _post(url: str, key: str, payload: dict[str, Any]) -> httpx.Response:
        for attempt in (1, 2):
            _wait_for_request_slot()
            retry_after = None
            try:
                reply = httpx.post(
                    url,
                    json=payload,
                    headers={"api-key": key},
                    timeout=settings.retry.http_timeout_seconds,
                )
            except httpx.TransportError as exc:
                failure = f"unreachable ({type(exc).__name__})"
            else:
                if reply.is_success:
                    return reply
                failure = f"HTTP {reply.status_code}"
                if reply.status_code != 429 and reply.status_code < 500:
                    raise EmbeddingUnavailableError(f"Embedding provider failed: {failure}.")
                retry_after = reply.headers.get("Retry-After")
            if attempt == 2:
                raise EmbeddingUnavailableError(f"Embedding provider failed: {failure}.")
            time.sleep(_retry_wait_seconds(retry_after))
        raise AssertionError("unreachable")

    @staticmethod
    def _parse_vectors(reply: httpx.Response, count: int, dimensions: int) -> list[list[float]]:
        try:
            vectors = [[float(value) for value in vector] for vector in reply.json()["embeddings"]["float"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingUnavailableError("Embedding reply was not the expected JSON.") from exc
        if len(vectors) != count or any(len(vector) != dimensions for vector in vectors):
            raise EmbeddingUnavailableError(
                "Embedding reply did not hold one vector of the configured size per text."
            )
        return vectors


# Langchain Embeddings (for ChromaDB)

def get_embeddings() -> Embeddings:
    """
    Get the Langchain-compatible embeddings for the configured arm.

    The arm (foundry_cohere, bedrock or azure_openai), model and dimensions come from
    settings.embedding. Every client has a request timeout and bounded retries.

    Example:
        embeddings = get_embeddings()

        # Embed documents
        vectos = embeddings.embed_documents(["text1", "text2"])

        # Embed query
        query_vector = embeddings.embed_query("search query")
    """
    arm = settings.embedding
    logger.info(f"Creating embeddings ({arm.provider}, model={arm.model})")

    if arm.provider == "bedrock":
        config = Config(
            connect_timeout=settings.retry.http_timeout_seconds,
            read_timeout=settings.retry.http_timeout_seconds,
            retries={"max_attempts": settings.retry.max_attempts, "mode": "standard"},
        )
        token = settings.llm.aws_bearer_token_bedrock
        client = None
        if token and token.get_secret_value().strip():
            client = create_aws_client(
                service_name="bedrock-runtime",
                region_name=settings.llm.aws_region,
                config=config,
                api_key=token,
            )
        return BedrockEmbeddings(
            model_id=arm.model,
            dimensions=arm.dimensions,
            region_name=settings.llm.aws_region,
            config=config,
            client=client,
        )

    if arm.provider == "azure_openai":
        foundry = settings.llm.azure_deployment(arm.model)
        if foundry is None:
            raise ValueError("EMBEDDING_PROVIDER=azure_openai needs AZURE_OPENAI_ENDPOINT and AZURE_API_KEY")
        base_url, deployment, key = foundry
        return OpenAIEmbeddings(
            model=deployment,
            dimensions=arm.dimensions,
            base_url=base_url,
            api_key=key,
            check_embedding_ctx_length=False,
            timeout=settings.retry.http_timeout_seconds,
            max_retries=settings.retry.max_attempts,
        )

    return FoundryCohereEmbeddings()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts synchronously."""
    return get_embeddings().embed_documents(texts)

def embed_query(query: str) -> list[float]:
    """Embed a search query synchronously."""
    return get_embeddings().embed_query(query)
