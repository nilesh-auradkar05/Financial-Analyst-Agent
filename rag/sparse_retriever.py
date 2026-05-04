"""Sparse retrieval helpers for SEC Filing chunks.

This module implements a lightweight BM25-style sparse retrieval.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any, Optional, Protocol, cast

from pydantic import BaseModel, Field

from rag.vector_store import IndexDocument, RetrievedChunk, SearchFilters, SearchResult


class SparseSearchConfig(BaseModel):
    """Configuration for BM-25 style sparse search."""

    top_k: int = Field(default=8, ge=1)
    candidate_pool_size: int = Field(default=512, ge=1)
    batch_size: int = Field(default=256, ge=1)
    k1: float = Field(default=1.5, gt=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    min_score: float = Field(default=0.0, ge=0.0)

class SparseSearchResult(BaseModel):
    """Debuggable sparse-search result before conversion to a retrieval chunk."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int

class SupportsSparseDocuments(Protocol):
    """Optional store protocol for direct sparse corpus access."""

    def iter_sparse_documents(
        self,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None,
    ) -> Iterable[IndexDocument]:
        ...

def tokenize_for_sparse_search(value: str) -> list[str]:
    """Normalize text into lexical tokens for BM25 scoring."""

    normalized = value.lower().replace("&", " and ")
    normalized = normalized.replace("'", "").replace("\u2019", "")
    return re.findall(r"[a-z0-9]+", normalized)

def _copy_model(model: BaseModel, update: dict[str, Any]) -> BaseModel:
    if hasattr(model, "model_copy"):
        return cast(BaseModel, model.model_copy(update=update))
    return cast(BaseModel, model.copy(update=update))

def _to_backend_filter(filters: Optional[SearchFilters]) -> Optional[dict[str, Any]]:
    return filters.to_backend_filter() if filters else None

def _document_from_parts(
    *,
    doc_id: str,
    text: str,
    metadata: Optional[dict[str, Any]],
) -> Optional[IndexDocument]:
    clean_text = (text or "").strip()
    if not clean_text:
        return None
    return IndexDocument(
        id=str(doc_id),
        text=clean_text,
        metadata=metadata or {},
    )

def _iter_sparse_protocol_documents(
    store: Any,
    *,
    filters: Optional[SearchFilters],
    limit: int,
) -> list[IndexDocument]:
    iterator = getattr(store, "iter_sparse_documents", None)
    if not callable(iterator):
        return []
    return list(cast(SupportsSparseDocuments, store).iter_sparse_documents(filters, limit))[:limit]

def _iter_chroma_documents(
    store: Any,
    *,
    filters: Optional[SearchFilters],
    limit: int,
) -> list[IndexDocument]:
    collection = getattr(store, "_collection", None)
    if collection is None:
        return []

    backend_filter = _to_backend_filter(filters)
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if backend_filter:
        get_kwargs["where"] = backend_filter
    if limit:
        get_kwargs["limit"] = limit

    try:
        raw = cast(dict[str, Any], collection.get(**get_kwargs))
    except TypeError:
        get_kwargs.pop("limit", None)
        raw = cast(dict[str, Any], collection.get(**get_kwargs))

    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    out: list[IndexDocument] = []
    for index, doc_id in enumerate(ids[:limit]):
        document = _document_from_parts(
            doc_id=str(doc_id),
            text=documents[index] if index < len(documents) else "",
            metadata=metadatas[index] if index < len(metadatas) else {},
        )
        if document is not None:
            out.append(document)
    return out

def _iter_qdrant_documents(
    store: Any,
    *,
    filters: Optional[SearchFilters],
    limit: int,
    batch_size: int,
) -> list[IndexDocument]:
    client = getattr(store, "client", None)
    collection_name = getattr(store, "collection_name", None)
    if client is None or not collection_name:
        return []

    filter_builder = getattr(store, '_to_qdrant_filter', None)
    query_filter = filter_builder(filters) if callable(filter_builder) else None

    out: list[IndexDocument] = []
    next_page: Any = None
    while len(out) < limit:
        points, next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=min(batch_size, limit - len(out)),
            with_payload=True,
            with_vectors=False,
            offset=next_page,
        )
        for point in points:
            payload = cast(dict[str, Any], getattr(point, "payload", None) or {})
            fallback_id = str(getattr(point, "id", ""))
            doc_id = str(payload.get("_retrieval_id") or fallback_id)
            text = str(payload.get("text") or "")
            metadata = {key: value for key, value in payload.items() if key != "text"}
            document = _document_from_parts(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            )
            if document is not None:
                out.append(document)
        if next_page is None:
            break
    return out[:limit]

def _iter_dense_fallback_documents(
    store: Any,
    *,
    query: str,
    filters: Optional[SearchFilters],
    limit: int,
) -> list[IndexDocument]:
    search = getattr(store, "search", None)
    if not callable(search):
        return []
    try:
        result = cast(SearchResult, search(query, filters=filters, n_results=limit))
    except TypeError:
        result = cast(SearchResult, search(query, filters=filters, top_k=limit))
    return [
        IndexDocument(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata or {}),
        )
        for chunk in result.chunks
    ]

def load_sparse_documents(
    store: Any,
    *,
    query: str,
    filters: Optional[SearchFilters],
    config: SparseSearchConfig,
) -> list[IndexDocument]:
    """Load candidate documents from a retrieval backend for sparse scoring."""

    limit = config.candidate_pool_size
    protocol_documents = _iter_sparse_protocol_documents(store, filters=filters, limit=limit)
    if protocol_documents:
        return protocol_documents

    chroma_documents = _iter_chroma_documents(store, filters=filters, limit=limit)
    if chroma_documents:
        return chroma_documents

    qdrant_documents = _iter_qdrant_documents(
        store,
        filters=filters,
        limit=limit,
        batch_size=config.batch_size,
    )
    if qdrant_documents:
        return qdrant_documents

    return _iter_dense_fallback_documents(
        store,
        query=query,
        filters=filters,
        limit=limit,
    )

def _bm25_scores(
    *,
    query_tokens: list[str],
    documents: list[IndexDocument],
    config: SparseSearchConfig,
) -> dict[str, float]:
    tokenized_documents = [tokenize_for_sparse_search(doc.text) for doc in documents]
    document_count = len(tokenized_documents)
    if document_count == 0 or not query_tokens:
        return {}

    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = sum(document_lengths) / document_count if document_count else 0.0
    document_frequencies: Counter[str] = Counter()

    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))

    scores: dict[str, float] = {}
    for document, tokens, document_length in zip(documents, tokenized_documents, document_lengths):
        term_counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            term_frequency = term_counts.get(token, 0)
            if term_frequency == 0:
                continue
            document_frequency = document_frequencies[token]
            idf = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + config.k1 * (
                1.0 - config.b + config.b * document_length / max(average_length, 1.0)
            )
            score += idf * (term_frequency * (config.k1 + 1.0)) / denominator
        if score >= config.min_score:
            scores[document.id] = score
    return scores


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values(), default=0.0)
    if max_score <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def search_sparse(
    store: Any,
    *,
    query: str,
    filters: Optional[SearchFilters] = None,
    top_k: int = 20,
    config: Optional[SparseSearchConfig] = None,
) -> SearchResult:
    """Run BM25-style sparse retrieval over SEC filing chunks."""

    started = time.time()
    resolved_config = config or SparseSearchConfig(top_k=top_k)
    resolved_config = cast(SparseSearchConfig, _copy_model(resolved_config, {"top_k": top_k}))
    query_tokens = tokenize_for_sparse_search(query)
    documents = load_sparse_documents(
        store,
        query=query,
        filters=filters,
        config=resolved_config,
    )
    raw_scores = _bm25_scores(
        query_tokens=query_tokens,
        documents=documents,
        config=resolved_config,
    )
    normalized_scores = _normalize_scores(raw_scores)
    documents_by_id = {document.id: document for document in documents}

    ranked_results: list[SparseSearchResult] = []
    for rank, (chunk_id, score) in enumerate(
        sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k],
        start=1,
    ):
        document = documents_by_id[chunk_id]
        ranked_results.append(
            SparseSearchResult(
                chunk_id=chunk_id,
                text=document.text,
                metadata=dict(document.metadata),
                score=round(score, 6),
                rank=rank,
            )
        )

    chunks: list[RetrievedChunk] = []
    for item in ranked_results:
        metadata = dict(item.metadata)
        metadata["sparse_score"] = item.score
        metadata["sparse_rank"] = item.rank
        normalized_score = normalized_scores.get(item.chunk_id, 0.0)
        chunks.append(
            RetrievedChunk(
                id=item.chunk_id,
                text=item.text,
                metadata=metadata,
                distance=max(0.0, 1.0 - normalized_score),
            )
        )

    return SearchResult(
        query=query,
        chunks=chunks,
        total_results=len(chunks),
        search_time_ms=(time.time() - started) * 1000,
        filter_used={
            "mode": "sparse",
            "filter": _to_backend_filter(filters),
            "candidate_pool_size": resolved_config.candidate_pool_size,
        },
    )
