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
from collections import deque
from collections.abc import Iterator
from typing import Any

import httpx
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from langchain_aws.utils import create_aws_client
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.config import settings

# Cohere Embed v5 fast on Foundry allows 20 requests and 20,000 tokens per minute; stay under both.
FOUNDRY_COHERE_MIN_INTERVAL_SECONDS = 60 / 18
FOUNDRY_COHERE_TOKENS_PER_MINUTE = 18_000
FOUNDRY_COHERE_MAX_REQUEST_TOKENS = 6_000
# The provider counts a rolling 60 s from its own clock; one second of slack covers network latency.
_FOUNDRY_COHERE_WINDOW_SECONDS = 61.0
_FOUNDRY_COHERE_MAX_TEXTS = 96
_FOUNDRY_COHERE_MAX_RETRY_WAIT_SECONDS = 60.0

# ponytail: per-process limiter, and tokens are estimated as ceil(chars / 4), which can
# under-count dense text (the retry covers small misses). Several processes sharing one
# deployment need a shared limiter (S7.9 cache store); provider token counts are the upgrade.
_pacing_lock = threading.Lock()
_recent_starts: deque[tuple[float, int]] = deque()  # (start time, estimated tokens)


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


def estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)


def _drop_expired_starts(now: float) -> None:
    while _recent_starts and _recent_starts[0][0] <= now - _FOUNDRY_COHERE_WINDOW_SECONDS:
        _recent_starts.popleft()


def _wait_for_request_slot(tokens: int) -> None:
    """Block until a request of `tokens` may start, then claim the slot; an idle process is not delayed."""
    with _pacing_lock:
        now = time.monotonic()
        _drop_expired_starts(now)
        wait = 0.0
        if _recent_starts:
            wait = _recent_starts[-1][0] + FOUNDRY_COHERE_MIN_INTERVAL_SECONDS - now
            in_window = sum(started_tokens for _, started_tokens in _recent_starts)
            for started, started_tokens in _recent_starts:
                if in_window + tokens <= FOUNDRY_COHERE_TOKENS_PER_MINUTE:
                    break
                in_window -= started_tokens
                wait = max(wait, started + _FOUNDRY_COHERE_WINDOW_SECONDS - now)
        if wait > 0:
            time.sleep(wait)
        now = time.monotonic()
        _drop_expired_starts(now)
        _recent_starts.append((now, tokens))


def _batches(texts: list[str]) -> Iterator[list[str]]:
    """Consecutive batches of at most 96 texts and 6,000 estimated tokens; an oversized text goes alone."""
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        tokens = estimate_tokens(text)
        if batch and (
            len(batch) == _FOUNDRY_COHERE_MAX_TEXTS
            or batch_tokens + tokens > FOUNDRY_COHERE_MAX_REQUEST_TOKENS
        ):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += tokens
    if batch:
        yield batch


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
        for batch in _batches(texts):
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
        tokens = sum(estimate_tokens(text) for text in payload["texts"])
        for attempt in (1, 2):
            _wait_for_request_slot(tokens)
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
