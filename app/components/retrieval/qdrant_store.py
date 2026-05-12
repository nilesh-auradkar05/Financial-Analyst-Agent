"""
Qdrant retrieval backend.

This module implements the same RetrievalStore contract as the Chroma backend,
while keeping Qdrant-specific point IDs, payload filters, and collection setup
encapsulated in one adapter.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional, cast

from loguru import logger
from qdrant_client import QdrantClient, models

from app.config import settings
from app.components.retrieval.embeddings import get_embeddings
from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievalStore,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
    _sanitize_metadata,
    canonical_section_display_name,
    canonical_section_key,
)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_VECTOR_DISTANCE = models.Distance.COSINE
PAYLOAD_INDEX_FIELDS = ("ticker", "filing_type", "section_key", "filing_date")

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)

def _to_point_id(document_id: str) -> str:
    """Convert project chunk IDs into stable Qdrant-compatible UUID point IDs.

    Qdrant accepts unsigned integers or UUIDs as point IDS.
    """
    try:
        return str(uuid.UUID(document_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, document_id))

def _payload_document_id(payload: dict[str, Any], fallback: str) -> str:
    value = payload.get("_retrieval_id")
    return str(value) if value else fallback

class QdrantVectorStore(RetrievalStore):
    """Qdrant-backed implementation of the retrieval contract.

    Qdrant point IDS must be UUIDs or unsigned integers, while this project uses
    semantic chunk IDs such as `AAPL_10-K_2025-10-31_business_000`. The adapter
    stores the project chunk ID in payload under `_retrieval_id` and uses a
    deterministic UUIDv5 as the Qdrant point ID. Higher layers continue seeing
    the original Chunk IDs.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: Optional[bool] = None,
        vector_size: Optional[int] = None,
    ):
        default_collection_name: str = settings.chroma.collection_name or "sec_filings"
        env_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.collection_name: str = collection_name or env_collection_name or default_collection_name
        self.url = url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.prefer_grpc = prefer_grpc if prefer_grpc is not None else _env_bool(
            "QDRANT_PREFER_GRPC",
            default=False,
        )
        self._configured_vector_size = vector_size or _env_int("QDRANT_VECTOR_SIZE")
        self._embeddings = get_embeddings()
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
        )
        self._collection_ready = False
        logger.info(f"Qdrant initialized (url={self.url}, collection={self.collection_name})")

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def count(self) -> int:
        return self.count_documents()

    def add_documents(self, documents: list[IndexDocument]) -> int:
        """Embed and upsert the documents into Qdrant."""
        normalized_documents = [
            IndexDocument(
                id=doc.id,
                text=doc.text,
                metadata=_sanitize_metadata(doc.metadata),
            )
            for doc in documents
            if doc.text.strip()
        ]
        if not normalized_documents:
            return 0

        texts = [doc.text for doc in normalized_documents]
        embeddings = self._embeddings.embed_documents(texts)
        if not embeddings:
            return 0

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size=vector_size)

        points = []
        for doc, vector in zip(normalized_documents, embeddings):
            payload = dict(doc.metadata)
            payload["_retrieval_id"] = doc.id
            payload["text"] = doc.text
            points.append(
                models.PointStruct(
                    id=_to_point_id(doc.id),
                    vector=cast(list[float], vector),
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} documents into Qdrant")
        return len(points)

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        """Search Qdrant using dense embeddings and optional metadata filters."""
        start_time = time.time()
        if not self._collection_exists():
            return SearchResult(
                query=query,
                chunks=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000,
                filter_used=self._filters_for_reporting(filters),
            )

        query_embedding = self._embeddings.embed_query(query)
        query_filter = self._to_qdrant_filter(filters)

        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=cast(list[float], query_embedding),
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.exception(
                f"Qdrant search failed | query={query!r} | filter={query_filter} | collection={self.collection_name}",
            )
            raise RuntimeError(
                f"Qdrant search failed for query={query!r}: {type(exc).__name__}: {exc}"
            ) from exc

        points = getattr(response, "points", response)
        chunks: list[RetrievedChunk] = []
        for point in points:
            payload = cast(dict[str, Any], getattr(point, "payload", None) or {})
            score = float(getattr(point, "score", 0.0) or 0.0)
            qdrant_id = str(getattr(point, "id", ""))
            chunks.append(
                RetrievedChunk(
                    id=_payload_document_id(payload, fallback=qdrant_id),
                    text=str(payload.get("text", "")),
                    metadata={k: v for k, v in payload.items() if k not in {"text"}},
                    distance=max(0.0, 1.0-score),
                )
            )

        search_time = (time.time() - start_time) * 1000
        return SearchResult(
            query=query,
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=search_time,
            filter_used=self._filters_for_reporting(filters),
        )

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        n_results: int = 8,
        query: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> SearchResult:
        """Search multiple SEC sections and return deduplicated results."""
        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        total_search_time_ms = 0.0

        for section_name in sections:
            section_query = query or f"{ticker} {section_name}"
            result, section_time_ms = self._search_section_with_fallback(
                ticker=ticker,
                section_name=section_name,
                query=section_query,
                n_results=n_results,
                filing_type=filing_type,
            )
            total_search_time_ms += section_time_ms
            for chunk in result.chunks:
                if chunk.id in seen_ids:
                    continue
                seen_ids.add(chunk.id)
                all_chunks.append(chunk)

        all_chunks.sort(key=lambda chunk: chunk.relevance_score, reverse=True)
        trimmed = all_chunks[:n_results]
        return SearchResult(
            query=query or f"{ticker} sections",
            chunks=trimmed,
            total_results=len(trimmed),
            search_time_ms=total_search_time_ms,
            filter_used={
                "ticker": ticker.upper(),
                "sections": sections,
                "filing_type": filing_type,
            },
        )

    def search_by_ticker(
        self,
        query: str,
        ticker: str,
        n_results: int = 5,
        section: Optional[str] = None,
    ) -> SearchResult:
        """Search documents for a ticker, optionally using section fallback logic."""
        if not section:
            return self.search(
                query=query,
                filters=SearchFilters(ticker=ticker),
                n_results=n_results,
            )

        result, _ = self._search_section_with_fallback(
            ticker=ticker,
            section_name=section,
            query=query,
            n_results=n_results,
        )
        return result

    def delete_by_ticker(self, ticker: str) -> int:
        """Delete all Qdrant points for a ticker"""
        ticker = ticker.upper().strip()
        if not self._collection_exists():
            return 0

        query_filter = self._to_qdrant_filter(SearchFilters(ticker=ticker))
        point_ids: list[Any] = []
        next_page: Any = None

        while True:
            points, next_page = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                with_payload=False,
                with_vectors=False,
                offset=next_page,
            )
            point_ids.extend(point.id for point in points)
            if next_page is None:
                break

        if not point_ids:
            return 0

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        logger.info(f"Deleted {len(point_ids)} Qdrant documents for {ticker}")
        return len(point_ids)

    def count_documents(self, filters: Optional[SearchFilters] = None) -> int:
        """Count documents using optional Qdrant payload filters."""
        if not self._collection_exists():
            return 0

        query_filter = self._to_qdrant_filter(filters)
        response = self._client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(response.count)

    def get_stats(self) -> dict[str, Any]:
        """Return backend stats in the same high-level shape as chroma."""
        if not self._collection_exists():
            return {
                "backend": "qdrant",
                "collection_name": self.collection_name,
                "total_documents": 0,
                "url": self.url,
                "status": "missing",
            }
        info = self._client.get_collection(collection_name=self.collection_name)

        return {
            "backend": "qdrant",
            "collection_name": self.collection_name,
            "total_documents": self.count_documents(),
            "url": self.url,
            "status": str(getattr(info, "status", "unknown")),
        }

    def create_payload_indexes(self) -> None:
        """Create metadata indexes used by financial filing filters"""
        for field_name in PAYLOAD_INDEX_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "already indexed" in message:
                    logger.debug(
                        f"Qdrant payload index already exists | field={field_name}",
                    )
                    continue
                raise RuntimeError(
                    f"Failed to create Qdrant payload index for field={field_name!r}."
                ) from exc

    def _search_section_with_fallback(
        self,
        *,
        ticker: str,
        section_name: str,
        query: str,
        n_results: int,
        filing_type: Optional[str] = None,
    ) -> tuple[SearchResult, float]:
        total_search_time_ms = 0.0
        section_key = canonical_section_key(section_name)
        legacy_section_name = canonical_section_display_name(section_name) or section_name

        primary_filters = (
            SearchFilters(ticker=ticker, filing_type=filing_type, section_key=section_key)
            if section_key
            else SearchFilters(
                ticker=ticker,
                filing_type=filing_type,
                section_name=legacy_section_name,
            )
        )
        primary_result = self.search(query=query, filters=primary_filters, n_results=n_results)
        total_search_time_ms += primary_result.search_time_ms
        if primary_result.has_results or not section_key:
            return primary_result, total_search_time_ms

        fallback_candidates = [legacy_section_name]
        if section_name not in fallback_candidates:
            fallback_candidates.append(section_name)

        seen_filter_names: set[str] = set()
        for candidate in fallback_candidates:
            if candidate in seen_filter_names:
                continue
            seen_filter_names.add(candidate)
            fallback_result = self.search(
                query=query,
                filters=SearchFilters(
                    ticker=ticker,
                    filing_type=filing_type,
                    section_name=candidate,
                ),
                n_results=n_results,
            )
            total_search_time_ms += fallback_result.search_time_ms
            if fallback_result.has_results:
                return fallback_result, total_search_time_ms

        return primary_result, total_search_time_ms

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready and self._collection_exists():
            return

        if not self._collection_exists():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=DEFAULT_VECTOR_DISTANCE,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' with vector_size={vector_size}",
            )

        self.create_payload_indexes()
        self._collection_ready = True

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.collection_name))
        except Exception:
            return False

    def _get_vector_size(self) -> int:
        if self._configured_vector_size:
            return self._configured_vector_size
        probe_embedding = self._embeddings.embed_query("dimension probe")
        self._configured_vector_size = len(probe_embedding)
        return self._configured_vector_size

    def _to_qdrant_filter(self, filters: Optional[SearchFilters]) -> Optional[models.Filter]:
        if filters is None:
            return None

        conditions: list[models.Condition] = []
        if filters.ticker:
            conditions.append(self._match_condition("ticker", filters.ticker.upper()))
        if filters.filing_type:
            conditions.append(self._match_condition("filing_type", filters.filing_type))
        if filters.section_key:
            conditions.append(self._match_condition("section_key", filters.section_key))
        elif filters.section_name:
            conditions.append(self._match_condition("section", filters.section_name))
        if filters.filing_date:
            conditions.append(self._match_condition("filing_date", filters.filing_date))
        for key, value in filters.extra.items():
            conditions.append(self._match_condition(key, value))

        if not conditions:
            return None
        return models.Filter(must=conditions)

    @staticmethod
    def _match_condition(key: str, value: Any) -> models.FieldCondition:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )

    @staticmethod
    def _filters_for_reporting(filters: Optional[SearchFilters]) -> Optional[dict[str, Any]]:
        return filters.to_backend_filter() if filters else None
