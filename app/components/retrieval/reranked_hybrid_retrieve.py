"""Dense + BM25 candidate generation followed by cross-encoder reranking.

This module implements the standard two-stage retrieval stack:

1. Generate a broad candidate set with dense vector search and BM25 lexical
   search.
2. Deduplicate candidates while preserving dense/BM25 provenance.
3. Score query/document pairs with a sentence-transformers CrossEncoder.
4. Return top-k evidence chunks with explicit reranker metadata.

The module is intentionally strict for evaluation: if the cross-encoder cannot
run, it raises instead of silently falling back to non-reranked hybrid output.
Silent fallback is how mislabeled result files happen. Humanity has suffered
enough JSON fraud.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Optional, Protocol, Sequence, cast

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.components.retrieval.hybrid_retrieve import (
    LangChainHybridSearchConfig,
    _call_store_search,
    _chunk_to_document,
    _load_sparse_documents_from_store,
    _stable_chunk_id,
)
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult

try:
    from app.components.retrieval.section_intent import SectionIntent, infer_section_intent, normalize_query_text
except ImportError:
    SectionIntent = Any  # type: ignore[misc, assignment]

    def infer_section_intent(query: str, *, max_sections: int = 3) -> Any:  # type: ignore[misc, no-redef]
        del query, max_sections
        return None

    def normalize_query_text(value: str) -> str:  # type: ignore[no-redef]
        return value.strip().lower()


DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_DENSE_TOP_K = 30
DEFAULT_SPARSE_TOP_K = 30
DEFAULT_FINAL_TOP_K = 5
DEFAULT_MAX_DOCUMENT_CHARS = 4_000
DEFAULT_RETRIEVAL_METHOD = "dense_bm25_cross_encoder_rerank"


class CrossEncoderLike(Protocol):
    """Protocol for sentence-transformers CrossEncoder-compatible models."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Any:
        """Score query/document pairs; higher is better."""


class RerankedHybridSearchConfig(BaseModel):
    """Configuration for dense + BM25 candidate generation and reranking."""

    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(default=DEFAULT_DENSE_TOP_K, ge=1)
    sparse_top_k: int = Field(default=DEFAULT_SPARSE_TOP_K, ge=1)
    final_top_k: int = Field(default=DEFAULT_FINAL_TOP_K, ge=1)
    candidate_pool_size: int = Field(default=512, ge=1)
    qdrant_scroll_batch_size: int = Field(default=128, ge=1)
    reranker_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME",
            DEFAULT_RERANKER_MODEL,
        )
    )
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_device: Optional[str] = Field(default_factory=lambda: os.getenv("RERANKER_DEVICE"))
    max_document_chars: int = Field(default=DEFAULT_MAX_DOCUMENT_CHARS, ge=256)
    fail_open_to_dense: bool = False
    use_section_prior: bool = True
    section_prior_boost: float = Field(default=0.20, ge=0.0)
    keyword_prior_boost: float = Field(default=0.03, ge=0.0)
    max_keyword_prior_boost: float = Field(default=0.15, ge=0.0)

    @field_validator("final_top_k")
    @classmethod
    def _validate_final_top_k(cls, final_top_k: int, info: Any) -> int:
        dense_top_k = int(info.data.get("dense_top_k", DEFAULT_DENSE_TOP_K))
        sparse_top_k = int(info.data.get("sparse_top_k", DEFAULT_SPARSE_TOP_K))
        if final_top_k > dense_top_k + sparse_top_k:
            raise ValueError("final_top_k cannot exceed dense_top_k + sparse_top_k")
        return final_top_k


class RerankedCandidate(BaseModel):
    """Candidate document with retrieval provenance and reranker score."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    document: Document
    chunk_id: str
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    reranker_score: float = 0.0
    section_prior_score: float = 0.0
    keyword_prior_score: float = 0.0
    final_rerank_score: float = 0.0

    @property
    def best_candidate_rank(self) -> int:
        ranks = [rank for rank in [self.dense_rank, self.bm25_rank] if rank is not None]
        return min(ranks) if ranks else 10_000

    @property
    def sources(self) -> list[str]:
        sources: list[str] = []
        if self.dense_rank is not None:
            sources.append("dense")
        if self.bm25_rank is not None:
            sources.append("bm25")
        return sources


class RerankedHybridSearchUnavailableError(RuntimeError):
    """Raised when reranked hybrid search cannot be executed."""


def search_reranked_hybrid(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str] = None,
    top_k: int = DEFAULT_FINAL_TOP_K,
    config: Optional[RerankedHybridSearchConfig] = None,
    reranker: Optional[CrossEncoderLike] = None,
) -> SearchResult:
    """Run dense + BM25 retrieval, then cross-encoder rerank to top-k.

    Evaluation mode fails closed by default. If sentence-transformers is missing
    or the reranker fails, this function raises unless ``fail_open_to_dense`` is
    explicitly set. That keeps result files honest: reranked output must contain
    reranker scores.
    """

    start_time = time.time()
    cfg = config or RerankedHybridSearchConfig(final_top_k=top_k)
    if top_k != cfg.final_top_k:
        cfg = cfg.model_copy(update={"final_top_k": top_k})

    filters = SearchFilters(ticker=ticker, filing_type=filing_type)
    dense_result = _call_store_search(
        store,
        query=query,
        filters=filters,
        n_results=cfg.dense_top_k,
    )
    dense_documents = [
        _chunk_to_document(chunk, source="dense", rank=rank)
        for rank, chunk in enumerate(dense_result.chunks, start=1)
    ]

    sparse_documents = _load_sparse_documents_from_store(
        store,
        filters=filters,
        config=LangChainHybridSearchConfig(
            dense_top_k=cfg.dense_top_k,
            sparse_top_k=cfg.sparse_top_k,
            final_top_k=cfg.final_top_k,
            candidate_pool_size=cfg.candidate_pool_size,
            qdrant_scroll_batch_size=cfg.qdrant_scroll_batch_size,
        ),
    )
    bm25_documents = _search_bm25_documents(
        query=query,
        documents=sparse_documents,
        top_k=cfg.sparse_top_k,
    )

    candidates = _merge_candidates(
        dense_documents=dense_documents,
        bm25_documents=bm25_documents,
    )
    if not candidates:
        raise RerankedHybridSearchUnavailableError(
            "Reranked hybrid produced no dense or BM25 candidates."
        )

    intent = infer_section_intent(query) if cfg.use_section_prior else None

    try:
        scored_candidates = _rerank_candidates(
            query=query,
            candidates=candidates,
            config=cfg,
            reranker=reranker,
            intent=intent,
        )
    except Exception as exc:
        if not cfg.fail_open_to_dense:
            raise RerankedHybridSearchUnavailableError(
                "Cross-encoder reranking failed. Install sentence-transformers, "
                "fix the model/device, or choose a non-reranked retrieval mode."
            ) from exc
        logger.warning("Reranked hybrid failed; falling back to dense results: {}", exc)
        return SearchResult(
            query=query,
            chunks=_dense_fallback_chunks(dense_result.chunks[: cfg.final_top_k]),
            total_results=min(len(dense_result.chunks), cfg.final_top_k),
            search_time_ms=(time.time() - start_time) * 1000,
            filter_used={
                "mode": "reranked_hybrid_dense_fallback",
                "retrieval_method": "dense_fallback_no_cross_encoder",
                "ticker": ticker.upper(),
                "filing_type": filing_type,
                "error": str(exc),
            },
        )

    top_candidates = scored_candidates[: cfg.final_top_k]
    chunks = [
        _candidate_to_chunk(candidate, rank=rank)
        for rank, candidate in enumerate(top_candidates, start=1)
    ]

    return SearchResult(
        query=query,
        chunks=chunks,
        total_results=len(chunks),
        search_time_ms=(time.time() - start_time) * 1000,
        filter_used={
            "mode": "reranked_hybrid",
            "retrieval_method": DEFAULT_RETRIEVAL_METHOD,
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "dense_top_k": cfg.dense_top_k,
            "sparse_top_k": cfg.sparse_top_k,
            "final_top_k": cfg.final_top_k,
            "candidate_count": len(candidates),
            "sparse_corpus_size": len(sparse_documents),
            "reranker_model": cfg.reranker_model_name,
            "use_section_prior": cfg.use_section_prior,
            "section_prior_target_sections": _intent_sections(intent),
        },
    )


def _search_bm25_documents(
    *,
    query: str,
    documents: list[Document],
    top_k: int,
) -> list[Document]:
    if not documents:
        return []
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k
    return list(retriever.invoke(query))


def _merge_candidates(
    *,
    dense_documents: list[Document],
    bm25_documents: list[Document],
) -> list[RerankedCandidate]:
    by_chunk_id: dict[str, RerankedCandidate] = {}

    for rank, document in enumerate(dense_documents, start=1):
        chunk_id = _document_chunk_id(document)
        by_chunk_id[chunk_id] = RerankedCandidate(
            document=document,
            chunk_id=chunk_id,
            dense_rank=rank,
        )

    for rank, document in enumerate(bm25_documents, start=1):
        chunk_id = _document_chunk_id(document)
        existing = by_chunk_id.get(chunk_id)
        if existing is None:
            metadata = dict(document.metadata)
            metadata["bm25_rank"] = rank
            metadata["retrieval_source"] = "bm25"
            by_chunk_id[chunk_id] = RerankedCandidate(
                document=Document(page_content=document.page_content, metadata=metadata),
                chunk_id=chunk_id,
                bm25_rank=rank,
            )
            continue

        metadata = dict(existing.document.metadata)
        metadata["bm25_rank"] = rank
        metadata["retrieval_source"] = "dense+bm25"
        by_chunk_id[chunk_id] = existing.model_copy(
            update={
                "document": Document(
                    page_content=existing.document.page_content,
                    metadata=metadata,
                ),
                "bm25_rank": rank,
            }
        )

    return sorted(by_chunk_id.values(), key=lambda candidate: candidate.best_candidate_rank)


def _rerank_candidates(
    *,
    query: str,
    candidates: list[RerankedCandidate],
    config: RerankedHybridSearchConfig,
    reranker: Optional[CrossEncoderLike] = None,
    intent: Any = None,
) -> list[RerankedCandidate]:
    model = reranker or _get_cross_encoder(
        model_name=config.reranker_model_name,
        device=config.reranker_device,
    )
    pairs = [
        (query, _clip_document_text(candidate.document.page_content, config.max_document_chars))
        for candidate in candidates
    ]
    scores = model.predict(
        pairs,
        batch_size=config.reranker_batch_size,
        show_progress_bar=False,
    )
    score_values = [float(score) for score in list(scores)]
    if len(score_values) != len(candidates):
        raise ValueError(
            "Reranker returned a different number of scores than candidates: "
            f"{len(score_values)} != {len(candidates)}"
        )

    scored_candidates = []
    for candidate, score in zip(candidates, score_values):
        section_prior = _section_prior_score(candidate, intent, config)
        keyword_prior = _keyword_prior_score(candidate, intent, config)
        final_score = score + section_prior + keyword_prior
        scored_candidates.append(
            candidate.model_copy(
                update={
                    "reranker_score": score,
                    "section_prior_score": section_prior,
                    "keyword_prior_score": keyword_prior,
                    "final_rerank_score": final_score,
                }
            )
        )

    return sorted(
        scored_candidates,
        key=lambda candidate: (-candidate.final_rerank_score, candidate.best_candidate_rank),
    )


@lru_cache(maxsize=4)
def _get_cross_encoder(*, model_name: str, device: Optional[str]) -> CrossEncoderLike:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RerankedHybridSearchUnavailableError(
            "sentence-transformers is required for reranked hybrid retrieval. "
            "Install it or use a non-reranked retrieval mode."
        ) from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    return cast(CrossEncoderLike, CrossEncoder(model_name, **kwargs))


def _candidate_to_chunk(candidate: RerankedCandidate, *, rank: int) -> RetrievedChunk:
    metadata = dict(candidate.document.metadata)
    metadata["chunk_id"] = candidate.chunk_id
    metadata["hybrid_rank"] = rank
    metadata["reranker_rank"] = rank
    metadata["reranker_score"] = candidate.reranker_score
    metadata["section_prior_score"] = candidate.section_prior_score
    metadata["keyword_prior_score"] = candidate.keyword_prior_score
    metadata["final_rerank_score"] = candidate.final_rerank_score
    metadata["dense_rank"] = candidate.dense_rank
    metadata["bm25_rank"] = candidate.bm25_rank
    metadata["retrieval_sources"] = candidate.sources
    metadata["retrieval_method"] = DEFAULT_RETRIEVAL_METHOD

    return RetrievedChunk(
        id=candidate.chunk_id,
        text=candidate.document.page_content,
        metadata=metadata,
        distance=_rank_to_distance(rank),
    )


def _dense_fallback_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    fallback_chunks: list[RetrievedChunk] = []
    for rank, chunk in enumerate(chunks, start=1):
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        metadata["retrieval_method"] = "dense_fallback_no_cross_encoder"
        metadata["reranker_rank"] = None
        metadata["reranker_score"] = None
        fallback_chunks.append(
            RetrievedChunk(
                id=str(getattr(chunk, "id", "")),
                text=str(getattr(chunk, "text", "")),
                metadata=metadata,
                distance=_rank_to_distance(rank),
            )
        )
    return fallback_chunks


def _document_chunk_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id") or document.metadata.get("_retrieval_id") or "")
    if chunk_id:
        return chunk_id
    chunk_id = _stable_chunk_id(document.page_content, document.metadata)
    document.metadata["chunk_id"] = chunk_id
    return chunk_id


def _clip_document_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _rank_to_distance(rank: int) -> float:
    return max(0.0, float(rank - 1) / max(rank, 1))


def _section_prior_score(
    candidate: RerankedCandidate,
    intent: Any,
    config: RerankedHybridSearchConfig,
) -> float:
    if intent is None or not config.use_section_prior:
        return 0.0
    target_sections = set(getattr(intent, "target_sections", []) or [])
    if not target_sections:
        return 0.0
    metadata = dict(candidate.document.metadata)
    section_key = str(metadata.get("section_key") or metadata.get("section_name") or "")
    if section_key in target_sections:
        return config.section_prior_boost
    return 0.0


def _keyword_prior_score(
    candidate: RerankedCandidate,
    intent: Any,
    config: RerankedHybridSearchConfig,
) -> float:
    if intent is None or not config.use_section_prior:
        return 0.0
    matched_terms = list(getattr(intent, "matched_terms", []) or [])
    if not matched_terms:
        return 0.0
    document_text = normalize_query_text(candidate.document.page_content)
    matches = sum(1 for term in matched_terms if normalize_query_text(str(term)) in document_text)
    return min(config.max_keyword_prior_boost, matches * config.keyword_prior_boost)


def _intent_sections(intent: Any) -> list[str]:
    if intent is None:
        return []
    return list(getattr(intent, "target_sections", []) or [])
