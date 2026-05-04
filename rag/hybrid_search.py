"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from rag.section_aware_search import search_section_aware
from rag.sparse_retriever import SparseSearchConfig, search_sparse
from rag.vector_store import RetrievedChunk, SearchFilters, SearchResult


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid retrieval."""

    dense_candidate_k: int = Field(default=20, ge=1)
    sparse_candidate_k: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.25, ge=0.0)
    use_section_aware_dense: bool = True
    sparse_candidate_pool_size: int = Field(default=512, ge=1)


class HybridChunkScore(BaseModel):
    """Debuggable hybrid score components for one chunk."""

    chunk_id: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_rrf_score: float = 0.0
    sparse_rrf_score: float = 0.0
    final_rrf_score: float = 0.0


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return cast(dict[str, Any], model.model_dump())
    return cast(dict[str, Any], model.dict())


def _copy_model(model: BaseModel, update: dict[str, Any]) -> BaseModel:
    if hasattr(model, "model_copy"):
        return cast(BaseModel, model.model_copy(update=update))
    return cast(BaseModel, model.copy(update=update))


def _chunk_identity(chunk: RetrievedChunk) -> str:
    chunk_id = str(getattr(chunk, "id", "") or "").strip()
    if chunk_id:
        return chunk_id
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    metadata_chunk_id = str(metadata.get("chunk_id", "") or "").strip()
    if metadata_chunk_id:
        return metadata_chunk_id
    return str(hash(getattr(chunk, "text", "")))


def _call_dense_search(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str],
    config: HybridSearchConfig,
) -> SearchResult:
    if config.use_section_aware_dense:
        return search_section_aware(
            store,
            query=query,
            ticker=ticker,
            filing_type=filing_type,
            top_k=config.dense_candidate_k,
        )

    filters = SearchFilters(ticker=ticker, filing_type=filing_type)
    try:
        return cast(
            SearchResult,
            store.search(query, filters=filters, n_results=config.dense_candidate_k),
        )
    except TypeError:
        return cast(
            SearchResult,
            store.search(query, filters=filters, top_k=config.dense_candidate_k),
        )


def _rank_map(chunks: list[RetrievedChunk]) -> dict[str, int]:
    return {_chunk_identity(chunk): rank for rank, chunk in enumerate(chunks, start=1)}


def _reciprocal_rank_score(*, weight: float, rank: Optional[int], rrf_k: int) -> float:
    if rank is None:
        return 0.0
    return weight / (rrf_k + rank)


def _fuse_chunks(
    *,
    dense_chunks: list[RetrievedChunk],
    sparse_chunks: list[RetrievedChunk],
    config: HybridSearchConfig,
) -> list[RetrievedChunk]:
    chunk_by_id: dict[str, RetrievedChunk] = {}
    scores_by_id: dict[str, float] = defaultdict(float)
    dense_ranks = _rank_map(dense_chunks)
    sparse_ranks = _rank_map(sparse_chunks)

    for chunk in dense_chunks + sparse_chunks:
        chunk_id = _chunk_identity(chunk)
        if chunk_id not in chunk_by_id:
            chunk_by_id[chunk_id] = chunk

    for chunk_id in chunk_by_id:
        dense_score = _reciprocal_rank_score(
            weight=config.dense_weight,
            rank=dense_ranks.get(chunk_id),
            rrf_k=config.rrf_k,
        )
        sparse_score = _reciprocal_rank_score(
            weight=config.sparse_weight,
            rank=sparse_ranks.get(chunk_id),
            rrf_k=config.rrf_k,
        )
        scores_by_id[chunk_id] = dense_score + sparse_score

    ranked_chunk_ids = sorted(
        scores_by_id,
        key=lambda chunk_id: (-scores_by_id[chunk_id], chunk_id),
    )

    fused_chunks: list[RetrievedChunk] = []
    max_score = max(scores_by_id.values(), default=0.0)
    for chunk_id in ranked_chunk_ids[: config.final_top_k]:
        source_chunk = chunk_by_id[chunk_id]
        dense_score = _reciprocal_rank_score(
            weight=config.dense_weight,
            rank=dense_ranks.get(chunk_id),
            rrf_k=config.rrf_k,
        )
        sparse_score = _reciprocal_rank_score(
            weight=config.sparse_weight,
            rank=sparse_ranks.get(chunk_id),
            rrf_k=config.rrf_k,
        )
        score = HybridChunkScore(
            chunk_id=chunk_id,
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
            dense_rrf_score=round(dense_score, 8),
            sparse_rrf_score=round(sparse_score, 8),
            final_rrf_score=round(scores_by_id[chunk_id], 8),
        )
        metadata = dict(getattr(source_chunk, "metadata", {}) or {})
        metadata["hybrid_score"] = _dump_model(score)
        normalized_score = scores_by_id[chunk_id] / max_score if max_score > 0.0 else 0.0
        fused_chunks.append(
            RetrievedChunk(
                id=getattr(source_chunk, "id", chunk_id),
                text=getattr(source_chunk, "text", ""),
                metadata=metadata,
                distance=max(0.0, 1.0 - normalized_score),
            )
        )
    return fused_chunks


def search_hybrid(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str] = None,
    top_k: int = 5,
    config: Optional[HybridSearchConfig] = None,
) -> SearchResult:
    """Run dense + sparse retrieval and fuse candidates with RRF."""

    started = time.time()
    resolved_config = config or HybridSearchConfig(final_top_k=top_k)
    resolved_config = cast(HybridSearchConfig, _copy_model(resolved_config, {"final_top_k": top_k}))

    dense_result = _call_dense_search(
        store,
        query=query,
        ticker=ticker,
        filing_type=filing_type,
        config=resolved_config,
    )
    sparse_result = search_sparse(
        store,
        query=query,
        filters=SearchFilters(ticker=ticker, filing_type=filing_type),
        top_k=resolved_config.sparse_candidate_k,
        config=SparseSearchConfig(
            top_k=resolved_config.sparse_candidate_k,
            candidate_pool_size=resolved_config.sparse_candidate_pool_size,
        ),
    )

    fused_chunks = _fuse_chunks(
        dense_chunks=list(dense_result.chunks),
        sparse_chunks=list(sparse_result.chunks),
        config=resolved_config,
    )
    wall_clock_ms = (time.time() - started) * 1000

    return SearchResult(
        query=query,
        chunks=fused_chunks,
        total_results=len(fused_chunks),
        search_time_ms=max(
            wall_clock_ms,
            dense_result.search_time_ms + sparse_result.search_time_ms,
        ),
        filter_used={
            "mode": "hybrid",
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "dense_candidate_k": resolved_config.dense_candidate_k,
            "sparse_candidate_k": resolved_config.sparse_candidate_k,
            "rrf_k": resolved_config.rrf_k,
            "dense_weight": resolved_config.dense_weight,
            "sparse_weight": resolved_config.sparse_weight,
            "use_section_aware_dense": resolved_config.use_section_aware_dense,
            "dense_filter": dense_result.filter_used,
            "sparse_filter": sparse_result.filter_used,
        },
    )
