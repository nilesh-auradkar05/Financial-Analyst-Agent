"""LangChain-backed hybrid retrieval for SEC filing chunks.

This module keeps project-specific retrieval policy in this repository while
outsourcing generic sparse ranking and weighted Reciprocal Rank Fusion to
LangChain:

- dense retrieval: the configured project vector store, now usually Qdrant
- sparse retrieval: LangChain BM25Retriever over the filtered filing corpus
- fusion: LangChain EnsembleRetriever weighted RRF

The implementation intentionally does not use the earlier custom BM25/RRF code.
That code was educational. This one is maintainable, which is less romantic but
usually survives contact with tests.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any, Optional, Sequence, cast

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever  # type: ignore[no-redef]

from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_SPARSE_WEIGHT = 0.35
DEFAULT_RRF_C = 60


class LangChainHybridSearchConfig(BaseModel):
    """Configuration for LangChain BM25 + weighted RRF retrieval."""

    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(default=20, ge=1)
    sparse_top_k: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    candidate_pool_size: int = Field(default=512, ge=1)
    qdrant_scroll_batch_size: int = Field(default=128, ge=1)
    dense_weight: float = Field(default=DEFAULT_DENSE_WEIGHT, ge=0.0)
    sparse_weight: float = Field(default=DEFAULT_SPARSE_WEIGHT, ge=0.0)
    rrf_c: int = Field(default=DEFAULT_RRF_C, ge=1)

    @field_validator("sparse_weight")
    @classmethod
    def _validate_nonzero_weight_sum(cls, sparse_weight: float, info: Any) -> float:
        dense_weight = float(info.data.get("dense_weight", DEFAULT_DENSE_WEIGHT))
        if dense_weight == 0.0 and sparse_weight == 0.0:
            raise ValueError("At least one hybrid retriever weight must be non-zero")
        return sparse_weight


class DenseStoreRetriever(BaseRetriever):
    """Adapter from the project's dense vector-store contract to LangChain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any
    filters: Optional[SearchFilters] = None
    n_results: int = Field(default=20, ge=1)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        del run_manager
        result = _call_store_search(
            self.store,
            query=query,
            filters=self.filters,
            n_results=self.n_results,
        )
        return [
            _chunk_to_document(chunk, source="dense", rank=rank)
            for rank, chunk in enumerate(result.chunks, start=1)
        ]


class HybridSearchUnavailableError(RuntimeError):
    """Raised when hybrid retrieval cannot construct a sparse corpus."""


def search_langchain_hybrid(
    store: Any,
    *,
    query: str,
    ticker: str,
    filing_type: Optional[str] = None,
    top_k: int = 5,
    config: Optional[LangChainHybridSearchConfig] = None,
) -> SearchResult:
    """Run dense + BM25 hybrid retrieval with LangChain weighted RRF.

    The sparse corpus is loaded from the active store using the same ticker and
    filing filters as the dense retriever. In the current sprint, Qdrant is the
    expected default store, so the main corpus path uses Qdrant ``scroll``.
    Chroma and test doubles are supported as fallback paths to keep evaluation
    portable.
    """

    start_time = time.time()
    cfg = config or LangChainHybridSearchConfig(final_top_k=top_k)
    if top_k != cfg.final_top_k:
        cfg = cfg.model_copy(update={"final_top_k": top_k})

    filters = SearchFilters(ticker=ticker, filing_type=filing_type)
    sparse_documents = _load_sparse_documents_from_store(store, filters=filters, config=cfg)

    if not sparse_documents:
        logger.warning(
            "Hybrid retrieval sparse corpus is empty; falling back to dense search | "
            "ticker={} | filing_type={}",
            ticker,
            filing_type,
        )
        dense_result = _call_store_search(
            store,
            query=query,
            filters=filters,
            n_results=cfg.final_top_k,
        )
        return SearchResult(
            query=query,
            chunks=dense_result.chunks,
            total_results=len(dense_result.chunks),
            search_time_ms=(time.time() - start_time) * 1000,
            filter_used={
                "mode": "langchain_hybrid_dense_fallback",
                "ticker": ticker.upper(),
                "filing_type": filing_type,
            },
        )

    dense_retriever = DenseStoreRetriever(
        store=store,
        filters=filters,
        n_results=cfg.dense_top_k,
    )
    sparse_retriever = BM25Retriever.from_documents(
        sparse_documents,
        preprocess_func=_tokenize_for_sparse,
    )
    sparse_retriever.k = cfg.sparse_top_k

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[cfg.dense_weight, cfg.sparse_weight],
        c=cfg.rrf_c,
        id_key="chunk_id",
    )

    fused_documents = _rerank_by_sparse_overlap(
        query=query,
        documents=ensemble.invoke(query),
    )[: cfg.final_top_k]
    chunks = [
        _document_to_chunk(document, rank=rank)
        for rank, document in enumerate(fused_documents, start=1)
    ]

    return SearchResult(
        query=query,
        chunks=chunks,
        total_results=len(chunks),
        search_time_ms=(time.time() - start_time) * 1000,
        filter_used={
            "mode": "langchain_hybrid",
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "dense_top_k": cfg.dense_top_k,
            "sparse_top_k": cfg.sparse_top_k,
            "final_top_k": cfg.final_top_k,
            "dense_weight": cfg.dense_weight,
            "sparse_weight": cfg.sparse_weight,
            "rrf_c": cfg.rrf_c,
            "sparse_corpus_size": len(sparse_documents),
        },
    )


def _call_store_search(
    store: Any,
    *,
    query: str,
    filters: Optional[SearchFilters],
    n_results: int,
) -> SearchResult:
    try:
        return cast(SearchResult, store.search(query, filters=filters, n_results=n_results))
    except TypeError:
        return cast(SearchResult, store.search(query, filters=filters, top_k=n_results))


def _load_sparse_documents_from_store(
    store: Any,
    *,
    filters: SearchFilters,
    config: LangChainHybridSearchConfig,
) -> list[Document]:
    """Load a filtered corpus for BM25.

    Preferred order:
    1. Qdrant scroll against the default backend.
    2. Chroma collection get for local fallback runs.
    3. ``store.documents`` test-double hook.
    4. Dense-search fallback as a last resort.
    """

    if _looks_like_qdrant_store(store):
        return _load_qdrant_documents(store, filters=filters, config=config)

    if hasattr(store, "collection"):
        documents = _load_chroma_documents(store, filters=filters, limit=config.candidate_pool_size)
        if documents:
            return documents

    iter_documents = _load_iter_sparse_documents(store, filters=filters, limit=config.candidate_pool_size)
    if iter_documents:
        return iter_documents

    test_documents = _load_test_double_documents(store, filters=filters, limit=config.candidate_pool_size)
    if test_documents:
        return test_documents

    dense_result = _call_store_search(
        store,
        query=_fallback_sparse_seed_query(filters),
        filters=filters,
        n_results=config.candidate_pool_size,
    )
    return [
        _chunk_to_document(chunk, source="dense_sparse_seed", rank=rank)
        for rank, chunk in enumerate(dense_result.chunks, start=1)
    ]


def _looks_like_qdrant_store(store: Any) -> bool:
    return hasattr(store, "client") and hasattr(store, "collection_name")


def _load_qdrant_documents(
    store: Any,
    *,
    filters: SearchFilters,
    config: LangChainHybridSearchConfig,
) -> list[Document]:
    if hasattr(store, "_collection_exists") and not store._collection_exists():
        return []

    query_filter = _qdrant_filter(store, filters)
    documents: list[Document] = []
    next_page: Any = None

    while len(documents) < config.candidate_pool_size:
        points, next_page = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=query_filter,
            limit=min(
                config.qdrant_scroll_batch_size,
                config.candidate_pool_size - len(documents),
            ),
            with_payload=True,
            with_vectors=False,
            offset=next_page,
        )
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            document = _document_from_payload(
                payload,
                fallback_id=str(getattr(point, "id", "")),
                source="qdrant_scroll",
            )
            if document is not None:
                documents.append(document)
        if next_page is None:
            break

    return documents


def _qdrant_filter(store: Any, filters: SearchFilters) -> Any:
    to_qdrant_filter = getattr(store, "_to_qdrant_filter", None)
    if callable(to_qdrant_filter):
        return to_qdrant_filter(filters)
    return None


def _load_chroma_documents(store: Any, *, filters: SearchFilters, limit: int) -> list[Document]:
    try:
        collection = store.collection
        backend_filter = filters.to_backend_filter()
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"], "limit": limit}
        if backend_filter:
            kwargs["where"] = backend_filter
        results = collection.get(**kwargs)
    except Exception as exc:  # pragma: no cover - Chroma varies by version.
        logger.debug("Could not load BM25 corpus from Chroma: {}", exc)
        return []

    ids = list(results.get("ids") or [])
    texts = list(results.get("documents") or [])
    metadatas = list(results.get("metadatas") or [])
    documents: list[Document] = []
    for index, text in enumerate(texts):
        metadata = dict(metadatas[index] if index < len(metadatas) and metadatas[index] else {})
        chunk_id = str(ids[index] if index < len(ids) else metadata.get("chunk_id", ""))
        metadata["chunk_id"] = chunk_id
        metadata["retrieval_source"] = "chroma_get"
        if text:
            documents.append(Document(page_content=str(text), metadata=metadata))
    return documents


def _load_iter_sparse_documents(store: Any, *, filters: SearchFilters, limit: int) -> list[Document]:
    iter_sparse_documents = getattr(store, "iter_sparse_documents", None)
    if not callable(iter_sparse_documents):
        return []

    try:
        raw_documents = iter_sparse_documents(filters=filters, limit=limit)
    except TypeError:
        raw_documents = iter_sparse_documents(filters=filters)

    if not isinstance(raw_documents, Iterable):
        return []

    documents: list[Document] = []
    for item in raw_documents:
        if len(documents) >= limit:
            break

        if isinstance(item, Document):
            metadata = dict(item.metadata)
            text = item.page_content
            chunk_id = metadata.get("chunk_id")
        elif isinstance(item, IndexDocument):
            metadata = dict(item.metadata)
            text = item.text
            chunk_id = item.id
        else:
            metadata = dict(getattr(item, "metadata", {}) or {})
            text = str(getattr(item, "text", "") or "")
            chunk_id = getattr(item, "id", None)

        if not text or not _metadata_matches_filters(metadata, filters):
            continue

        metadata.setdefault("chunk_id", str(chunk_id or _stable_chunk_id(text, metadata)))
        metadata.setdefault("retrieval_source", "iter_sparse_documents")
        documents.append(Document(page_content=str(text), metadata=metadata))

    return documents


def _load_test_double_documents(store: Any, *, filters: SearchFilters, limit: int) -> list[Document]:
    raw_documents = getattr(store, "documents", None)
    if not raw_documents:
        return []

    documents: list[Document] = []
    for item in raw_documents[:limit]:
        if isinstance(item, Document):
            metadata = dict(item.metadata)
            if not _metadata_matches_filters(metadata, filters):
                continue
            metadata.setdefault("chunk_id", _stable_chunk_id(item.page_content, metadata))
            documents.append(Document(page_content=item.page_content, metadata=metadata))
            continue

        text = str(getattr(item, "text", "") or "")
        metadata = dict(getattr(item, "metadata", {}) or {})
        if not text or not _metadata_matches_filters(metadata, filters):
            continue
        metadata.setdefault("chunk_id", str(getattr(item, "id", "") or _stable_chunk_id(text, metadata)))
        metadata["retrieval_source"] = "test_double"
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _document_from_payload(
    payload: dict[str, Any],
    *,
    fallback_id: str,
    source: str,
) -> Optional[Document]:
    text = str(payload.get("text", "") or "")
    if not text.strip():
        return None
    chunk_id = str(
        payload.get("_retrieval_id")
        or payload.get("chunk_id")
        or payload.get("id")
        or fallback_id
    )
    metadata = {key: value for key, value in payload.items() if key != "text"}
    metadata["chunk_id"] = chunk_id
    metadata.setdefault("retrieval_source", source)
    return Document(page_content=text, metadata=metadata)


def _chunk_to_document(chunk: RetrievedChunk, *, source: str, rank: int) -> Document:
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    chunk_id = str(getattr(chunk, "id", "") or metadata.get("chunk_id") or "")
    if not chunk_id:
        chunk_id = _stable_chunk_id(getattr(chunk, "text", ""), metadata)
    metadata["chunk_id"] = chunk_id
    metadata["retrieval_source"] = source
    metadata[f"{source}_rank"] = rank
    metadata[f"{source}_distance"] = float(getattr(chunk, "distance", 0.0))
    return Document(page_content=str(getattr(chunk, "text", "")), metadata=metadata)


def _document_to_chunk(document: Document, *, rank: int) -> RetrievedChunk:
    metadata = dict(document.metadata)
    chunk_id = str(metadata.get("chunk_id") or _stable_chunk_id(document.page_content, metadata))
    metadata["hybrid_rank"] = rank
    metadata["retrieval_method"] = "langchain_bm25_ensemble_rrf"
    return RetrievedChunk(
        id=chunk_id,
        text=document.page_content,
        metadata=metadata,
        distance=_rank_to_distance(rank),
    )


def _tokenize_for_sparse(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _rerank_by_sparse_overlap(query: str, documents: list[Document]) -> list[Document]:
    query_tokens = set(_tokenize_for_sparse(query))
    if not query_tokens:
        return documents

    ranked_documents = [
        (
            _sparse_overlap_score(query_tokens, document.page_content),
            rank,
            document,
        )
        for rank, document in enumerate(documents)
    ]
    ranked_documents.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in ranked_documents]


def _sparse_overlap_score(query_tokens: set[str], text: str) -> float:
    document_tokens = set(_tokenize_for_sparse(text))
    if not document_tokens:
        return 0.0
    return sum(1 for token in query_tokens if token in document_tokens) / len(query_tokens)


def _rank_to_distance(rank: int) -> float:
    # The project contract exposes relevance as 1 / (1 + distance). This rank-based
    # synthetic distance preserves deterministic ordering after RRF.
    return max(0.0, float(rank - 1) / max(rank, 1))


def _metadata_matches_filters(metadata: dict[str, Any], filters: SearchFilters) -> bool:
    if filters.ticker and str(metadata.get("ticker", "")).upper() != filters.ticker.upper():
        return False
    if filters.filing_type and metadata.get("filing_type") != filters.filing_type:
        return False
    if filters.section_key and metadata.get("section_key") != filters.section_key:
        return False
    if filters.section_name:
        section = metadata.get("section_name") or metadata.get("section")
        if section != filters.section_name:
            return False
    if filters.filing_date and metadata.get("filing_date") != filters.filing_date:
        return False
    return all(metadata.get(key) == value for key, value in filters.extra.items())


def _stable_chunk_id(text: str, metadata: dict[str, Any]) -> str:
    ticker = str(metadata.get("ticker", "UNK"))
    section = str(metadata.get("section_key") or metadata.get("section_name") or "section")
    return f"{ticker}_{section}_{abs(hash(text))}"


def _fallback_sparse_seed_query(filters: SearchFilters) -> str:
    parts: Sequence[str] = [
        value
        for value in [filters.ticker, filters.filing_type, filters.section_key, filters.section_name]
        if value
    ]
    return " ".join(parts) or "SEC filing"
