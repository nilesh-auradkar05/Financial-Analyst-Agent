"""Section-aware retrieval helpers."""


from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from rag.section_intent import SectionIntent, infer_section_intent, normalize_query_text
from rag.sections import canonical_section_key
from rag.vector_store import RetrievedChunk, SearchFilters, SearchResult


class SectionAwareSearchConfig(BaseModel):
    """Turning parameters for section-aware retrieval."""

    targeted_candidate_k: int = Field(default=8, ge=1)
    fallback_candidate_k: int = Field(default=8, ge=1)
    section_match_boost: float = Field(default=0.35, ge=0.0)
    targeted_candidate_boost: float = Field(default=0.05, ge=0.0)
    keyword_match_boost: float = Field(default=0.03, ge=0.0)
    max_keyword_boost: float = Field(default=0.15, ge=0.0)

class SectionAwareChunkScore(BaseModel):
    """Debuggable score components for a reranked chunk."""

    chunk_id: str
    base_score: float
    section_boost: float
    targeted_boost: float
    keyword_boost: float
    final_score: float

def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return cast(dict[str, Any], model.model_dump())
    return cast(dict[str, Any], model.__dict__)

@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    """Private score container used during reranking."""

    chunk: RetrievedChunk
    score: SectionAwareChunkScore

def _chunk_identity(chunk: RetrievedChunk) -> str:
    chunk_id = str(getattr(chunk, "id", "") or "").strip()
    if chunk_id:
        return chunk_id
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    metadata_chunk_id = str(metadata.get("chunk_id") or "").strip()
    if metadata_chunk_id:
        return metadata_chunk_id
    return normalize_query_text(getattr(chunk, "text", ""))[:120]

def _chunk_section_key(chunk: RetrievedChunk) -> Optional[str]:
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    section_name = metadata.get("section_key") or metadata.get("section_name") or metadata.get("section")
    if not section_name:
        return None
    section_text = str(section_name)
    return canonical_section_key(section_text) or normalize_query_text(section_text).replace(" ", "_")

def _keyword_overlap_count(chunk: RetrievedChunk, matched_terms: list[str]) -> int:
    text = normalize_query_text(getattr(chunk, "text", ""))
    return sum(1 for term in matched_terms if normalize_query_text(term) in text)

def _call_search(
    store: Any,
    *,
    query: str,
    filters: SearchFilters,
    n_results: int,
) -> SearchResult:
    try:
        return cast(SearchResult, store.search(query, filters=filters, top_k=n_results))
    except TypeError:
        return cast(SearchResult, store.search(query, filters=filters, n_results=n_results))

def _call_search_sections(
    store: Any,
    *,
    ticker: str,
    sections: list[str],
    query: str,
    filing_type: Optional[str],
    n_results: int,
) -> SearchResult:
    try:
        return cast(
            SearchResult,
            store.search_sections(
                ticker=ticker,
                sections=sections,
                top_k=n_results,
                query=query,
                filing_type=filing_type,
            ),
        )
    except TypeError:
        return cast(
            SearchResult,
            store.search_sections(
                ticker=ticker,
                sections=sections,
                n_results=n_results,
                query=query,
                filing_type=filing_type,
            ),
        )

def _score_chunk(
    chunk: RetrievedChunk,
    *,
    intent: SectionIntent,
    targeted_candidate_ids: set[str],
    config: SectionAwareSearchConfig,
) -> SectionAwareChunkScore:
    chunk_id = _chunk_identity(chunk)
    base_score = float(getattr(chunk, "relevance_score", 0.0))
    section_key = _chunk_section_key(chunk)
    section_boost = (
        config.section_match_boost
        if section_key and section_key in set(intent.target_sections)
        else 0.0
    )
    targeted_boost = config.targeted_candidate_boost if chunk_id in targeted_candidate_ids else 0.0
    keyword_boost = min(
        config.max_keyword_boost,
        _keyword_overlap_count(chunk, intent.matched_terms) * config.keyword_match_boost,
    )
    final_score = base_score + section_boost + targeted_boost + keyword_boost

    return SectionAwareChunkScore(
        chunk_id=chunk_id,
        base_score=round(base_score, 6),
        section_boost=round(section_boost, 6),
        targeted_boost=round(targeted_boost, 6),
        keyword_boost=round(keyword_boost, 6),
        final_score=round(final_score, 6),
    )

def _rerank_chunks(
    chunks: list[RetrievedChunk],
    *,
    intent: SectionIntent,
    targeted_candidate_ids: set[str],
    config: SectionAwareSearchConfig,
) -> list[RetrievedChunk]:
    best_by_id: dict[str, _ScoredChunk] = {}

    for chunk in chunks:
        score = _score_chunk(
            chunk,
            intent=intent,
            targeted_candidate_ids=targeted_candidate_ids,
            config=config,
        )
        existing = best_by_id.get(score.chunk_id)
        if existing is None or score.final_score > existing.score.final_score:
            best_by_id[score.chunk_id] = _ScoredChunk(chunk=chunk, score=score)

    scored_chunks = sorted(
        best_by_id.values(),
        key=lambda item: (-item.score.final_score, item.score.chunk_id),
    )

    reranked: list[RetrievedChunk] = []
    for item in scored_chunks:
        chunk = cast(RetrievedChunk, item.chunk)
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        metadata["section_aware_score"] = _dump_model(item.score)
        reranked.append(
            RetrievedChunk(
                id=getattr(chunk, "id", ""),
                text=getattr(chunk, "text", ""),
                metadata=metadata,
                distance=float(getattr(chunk, "distance", 0.0)),
            )
        )

    return reranked

def search_section_aware(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str] = None,
    top_k: int = 5,
    intent: Optional[SectionIntent] = None,
    config: Optional[SectionAwareSearchConfig] = None,
) -> SearchResult:
    """Run section-aware retrieval and rerank merged candidates."""

    started = time.time()
    resolved_config = config or SectionAwareSearchConfig()
    resolved_intent = intent or infer_section_intent(query)

    fallback_result = _call_search(
        store,
        query=query,
        filters=SearchFilters(ticker=ticker, filing_type=filing_type),
        n_results=max(top_k, resolved_config.fallback_candidate_k),
    )

    targeted_result: Optional[SearchResult] = None
    targeted_candidate_ids: set[str] = set()
    candidate_chunks = list(fallback_result.chunks)
    total_search_time_ms = fallback_result.search_time_ms

    if resolved_intent.target_sections:
        targeted_result = _call_search_sections(
            store,
            ticker=ticker,
            sections=resolved_intent.target_sections,
            query=query,
            filing_type=filing_type,
            n_results=max(top_k, resolved_config.targeted_candidate_k),
        )
        targeted_candidate_ids = {_chunk_identity(chunk) for chunk in targeted_result.chunks}
        candidate_chunks = list(targeted_result.chunks) + candidate_chunks
        total_search_time_ms += targeted_result.search_time_ms

    reranked_chunks = _rerank_chunks(
        candidate_chunks,
        intent=resolved_intent,
        targeted_candidate_ids=targeted_candidate_ids,
        config=resolved_config,
    )[:top_k]

    wall_clock_ms = (time.time() - started) * 1000
    return SearchResult(
        query=query,
        chunks=reranked_chunks,
        total_results=len(reranked_chunks),
        search_time_ms=max(total_search_time_ms, wall_clock_ms),
        filter_used={
            "mode": "section_aware",
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "target_sections": resolved_intent.target_sections,
            "intent_confidence": resolved_intent.confidence,
            "matched_terms": resolved_intent.matched_terms,
            "targeted_search_used": targeted_result is not None,
        },
    )
