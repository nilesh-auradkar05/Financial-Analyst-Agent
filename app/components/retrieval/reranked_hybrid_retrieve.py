"""Rerank a dense or hybrid candidate pool with hosted Cohere Rerank on Microsoft Foundry.

Two-stage retrieval: the store's top `candidate_pool` (50 by default) under the ticker and
filing-type filters (dense `search` or fused `search_hybrid`), scored in one rerank call, best K kept. Fails closed: if the reranker
cannot run, it raises instead of returning unreranked chunks, because a silent fallback is how
mislabeled result files happen.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Literal, Optional, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, computed_field

from app.components.retrieval.embeddings import RequestPacer, estimate_tokens
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult
from app.config import settings

RERANK_CANDIDATE_POOL = 50
_MAX_DOCUMENT_CHARS = 4_000
_RERANK_TIMEOUT_SECONDS = 30.0
_RERANK_ATTEMPTS = 3
_RERANK_BACKOFF_SECONDS = 0.1
_GAP_RANK = 15

# The rerank deployment allows 50 requests and 50,000 tokens per minute (owner, 2026-11-23); stay
# under both, in a window of its own.
_rerank_pacer = RequestPacer(min_interval_seconds=60 / 45, tokens_per_minute=45_000)


class RerankerLike(Protocol):
    @property
    def identity(self) -> str:
        """Names the scoring model in result provenance."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, in input order; higher is better."""


class RerankedHybridSearchUnavailableError(RuntimeError):
    """Raised when reranked search cannot be executed."""


class TraceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section_key: str | None
    dense_rank: int
    dense_score: float
    rerank_rank: int
    rerank_score: float


class RetrievalTrace(BaseModel):
    """What one reranked search saw: its candidate pool in store order, rerank positions and
    the dense ranks it kept. Holds no chunk text."""

    model_config = ConfigDict(extra="forbid")

    query: str
    ticker: str
    filing_type: str | None
    candidate_source: Literal["dense", "hybrid"]
    candidate_pool: int
    candidates: list[TraceCandidate]
    kept_dense_ranks: list[int]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dense_score_gap(self) -> float | None:
        """Dense score at rank 1 minus at rank min(15, n); None below two candidates."""
        if len(self.candidates) < 2:
            return None
        reads = min(_GAP_RANK, len(self.candidates))
        return self.candidates[0].dense_score - self.candidates[reads - 1].dense_score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pool_sections(self) -> dict[str, int]:
        return dict(Counter(c.section_key or "unknown_section" for c in self.candidates))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class FoundryCohereReranker:
    """Cohere Rerank v5 fast on Microsoft Foundry (ADR-0011)."""

    @property
    def identity(self) -> str:
        return f"foundry:{(settings.llm.azure_foundry_rerank_model or '').strip()}"

    def score(self, query: str, documents: list[str]) -> list[float]:
        llm = settings.llm
        endpoint = (llm.azure_openai_endpoint or "").strip().rstrip("/")
        key = llm.azure_api_key.get_secret_value().strip() if llm.azure_api_key else ""
        deployment = (llm.azure_foundry_rerank_model or "").strip()
        if not (endpoint and key and deployment):
            raise RerankedHybridSearchUnavailableError(
                "Hosted rerank needs AZURE_OPENAI_ENDPOINT, AZURE_API_KEY and "
                "AZURE_FOUNDRY_RERANK_MODEL."
            )
        payload = {
            "model": deployment,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        reply = self._post(f"{endpoint}/providers/cohere/v2/rerank", key, payload)
        return self._parse_scores(reply, len(documents))

    @staticmethod
    def _post(url: str, key: str, payload: dict[str, Any]) -> httpx.Response:
        tokens = estimate_tokens(payload["query"]) + sum(
            estimate_tokens(text) for text in payload["documents"]
        )
        for attempt in range(1, _RERANK_ATTEMPTS + 1):
            _rerank_pacer.wait_for_slot(tokens)
            try:
                reply = httpx.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=_RERANK_TIMEOUT_SECONDS,
                )
                failure = f"HTTP {reply.status_code}"
                retryable = reply.status_code == 429 or reply.status_code >= 500
                if reply.is_success:
                    return reply
            except httpx.TransportError as exc:
                failure = f"unreachable ({type(exc).__name__})"
                retryable = True
            if not retryable or attempt == _RERANK_ATTEMPTS:
                raise RerankedHybridSearchUnavailableError(
                    f"Rerank provider failed: {failure}."
                )
            time.sleep(_RERANK_BACKOFF_SECONDS * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _parse_scores(reply: httpx.Response, count: int) -> list[float]:
        try:
            results = reply.json()["results"]
            scores: dict[int, float] = {}
            for item in results:
                index, score = item["index"], item["relevance_score"]
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < count
                    or index in scores
                    or not _is_number(score)
                ):
                    raise ValueError
                scores[index] = float(score)
        except (ValueError, KeyError, TypeError) as exc:
            raise RerankedHybridSearchUnavailableError(
                "Rerank reply was not the expected JSON."
            ) from exc
        if len(scores) != count:
            raise RerankedHybridSearchUnavailableError(
                "Rerank reply did not score every document exactly once."
            )
        return [scores[index] for index in range(count)]


def search_reranked(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str] = None,
    top_k: int = 10,
    candidates: Literal["dense", "hybrid"] = "dense",
    reranker: Optional[RerankerLike] = None,
    candidate_pool: int = RERANK_CANDIDATE_POOL,
) -> SearchResult:
    """Rerank the store's top `candidate_pool` dense or hybrid candidates and return the best `top_k`.

    Equal scores keep candidate order. Raises `RerankedHybridSearchUnavailableError` when the
    reranker is unconfigured or fails; no unreranked result is ever returned.
    """

    started = time.perf_counter()
    search = store.search if candidates == "dense" else store.search_hybrid
    pool: list[RetrievedChunk] = search(
        query,
        filters=SearchFilters(ticker=ticker, filing_type=filing_type),
        n_results=candidate_pool,
    ).chunks

    chunks: list[RetrievedChunk] = []
    scored: list[tuple[int, tuple[RetrievedChunk, float]]] = []
    ranked: list[tuple[int, tuple[RetrievedChunk, float]]] = []
    if pool:
        reranker = reranker or FoundryCohereReranker()
        scores = [float(score) for score in reranker.score(query, [c.text[:_MAX_DOCUMENT_CHARS] for c in pool])]
        if len(scores) != len(pool):
            raise RerankedHybridSearchUnavailableError(
                "Reranker returned a different number of scores than candidates."
            )
        scored = list(enumerate(zip(pool, scores), start=1))
        ranked = sorted(scored, key=lambda item: -item[1][1])
        chunks = [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "reranker_score": score,
                    "candidate_rank": rank,
                    "candidate_source": candidates,
                    "reranker_model": reranker.identity,
                },
                distance=chunk.distance,
            )
            for rank, (chunk, score) in ranked[:top_k]
        ]

    rerank_rank = {rank: place for place, (rank, _) in enumerate(ranked, start=1)}
    trace = RetrievalTrace(
        query=query,
        ticker=ticker,
        filing_type=filing_type,
        candidate_source=candidates,
        candidate_pool=candidate_pool,
        candidates=[
            TraceCandidate(
                chunk_id=chunk.id,
                section_key=chunk.metadata.get("section_key"),
                dense_rank=rank,
                dense_score=chunk.relevance_score,
                rerank_rank=rerank_rank[rank],
                rerank_score=score,
            )
            for rank, (chunk, score) in scored
        ],
        kept_dense_ranks=[chunk.metadata["candidate_rank"] for chunk in chunks],
    )
    return SearchResult(
        query=query,
        chunks=chunks,
        total_results=len(chunks),
        search_time_ms=(time.perf_counter() - started) * 1000,
        filter_used={
            "mode": f"{candidates}_rerank",
            "ticker": ticker.upper(),
            "filing_type": filing_type,
        },
        trace=trace,
    )
