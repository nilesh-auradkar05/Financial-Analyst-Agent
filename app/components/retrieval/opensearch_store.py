"""
OpenSearch retrieval backend.

Implements the RetrievalStore contract on an OpenSearch k-NN index. Chunk IDs are the
document `_id`, so higher layers see the same IDs they wrote. Only index create/exists,
`_bulk`, `_search` and `_count` are used, so the adapter also fits OpenSearch Serverless.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langsmith import get_current_run_tree
from loguru import logger
from opensearchpy import NotFoundError, OpenSearch, OpenSearchException, RequestError, helpers

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
from app.config import settings

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_TIMEOUT_S = 5.0
VECTOR_FIELD = "embedding"
REQUIRED_METADATA_KEYS = ("ticker", "filing_type", "filing_date", "accession_number", "section_key")
FILTER_FIELDS = frozenset(
    {
        "accession_number",
        "chunk_id",
        "ticker",
        "filing_type",
        "filing_date",
        "section_key",
        "section",
    }
)
DELETE_PAGE_SIZE = 500


class OpenSearchVectorStore(RetrievalStore):
    """OpenSearch-backed implementation of the retrieval contract.

    Vectors use cosine similarity on a Lucene HNSW index, which applies metadata filters
    inside the k-NN query, so a filtered search returns up to `n_results` matching chunks.
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        default_index_name: str = settings.chroma.collection_name or "sec_filings"
        self.index_name: str = (
            index_name or os.getenv("OPENSEARCH_INDEX_NAME") or default_index_name
        )
        self.url: str = url or os.getenv("OPENSEARCH_URL") or DEFAULT_OPENSEARCH_URL
        timeout = timeout_s or float(os.getenv("OPENSEARCH_TIMEOUT_S", DEFAULT_TIMEOUT_S))
        self._embeddings = embeddings or get_embeddings()
        self._embedding_model: str = str(getattr(self._embeddings, "model"))
        self._client = OpenSearch(
            hosts=[self.url],
            timeout=timeout,
            max_retries=0,
            retry_on_timeout=False,
        )
        logger.info(f"OpenSearch initialized (index={self.index_name})")

    @property
    def count(self) -> int:
        return self.count_documents()

    def healthcheck(self) -> bool:
        try:
            return bool(self._client.ping())
        except OpenSearchException:
            return False

    def add_documents(self, documents: list[IndexDocument]) -> int:
        """Embed and upsert the documents, keyed by chunk id."""
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

        for doc in normalized_documents:
            missing = [key for key in REQUIRED_METADATA_KEYS if not doc.metadata.get(key)]
            if missing:
                raise ValueError(f"Document {doc.id!r} is missing required metadata: {missing}")

        vectors = self._embeddings.embed_documents([doc.text for doc in normalized_documents])
        self._ensure_index(dimension=len(vectors[0]))

        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    **doc.metadata,
                    "text": doc.text,
                    "embedding_id": f"ollama:{self._embedding_model}:{len(vector)}",
                    VECTOR_FIELD: vector,
                },
            }
            for doc, vector in zip(normalized_documents, vectors)
        ]
        self._bulk(actions)
        logger.info(f"Upserted {len(actions)} documents into OpenSearch")
        return len(actions)

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        """Search by cosine similarity with exact-match metadata filters."""
        start_time = time.time()
        filter_query = self._to_filter_query(filters)
        chunks: list[RetrievedChunk] = []

        try:
            if run := get_current_run_tree():
                run.metadata.update(
                    embedding_model=self._embedding_model, embedding_provider="ollama"
                )
            knn: dict[str, Any] = {
                "vector": self._embeddings.embed_query(query),
                "k": n_results,
            }
            if filter_query:
                knn["filter"] = filter_query
            response = self._client.search(
                index=self.index_name,
                body={
                    "size": n_results,
                    "_source": {"excludes": [VECTOR_FIELD]},
                    "query": {"knn": {VECTOR_FIELD: knn}},
                },
            )
        except NotFoundError:
            response = {"hits": {"hits": []}}
        except OpenSearchException as exc:
            logger.exception(f"OpenSearch search failed | index={self.index_name}")
            raise RuntimeError(f"OpenSearch search failed: {type(exc).__name__}: {exc}") from exc

        for hit in response["hits"]["hits"]:
            source = dict(hit["_source"])
            text = str(source.pop("text", ""))
            chunks.append(
                RetrievedChunk(
                    id=hit["_id"],
                    text=text,
                    metadata=source,
                    # Cosine space score is (2 - d) / 2 with d = 1 - cos, so d = 2 * (1 - score).
                    distance=max(0.0, 2.0 * (1.0 - float(hit["_score"]))),
                )
            )

        return SearchResult(
            query=query,
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=(time.time() - start_time) * 1000,
            filter_used=filters.to_backend_filter() if filters else None,
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
        """Delete all documents for a ticker."""
        filter_query = self._to_filter_query(SearchFilters(ticker=ticker.upper().strip()))
        deleted = 0

        while True:
            try:
                response = self._client.search(
                    index=self.index_name,
                    body={"size": DELETE_PAGE_SIZE, "_source": False, "query": filter_query},
                )
            except NotFoundError:
                break
            ids = [hit["_id"] for hit in response["hits"]["hits"]]
            if not ids:
                break
            self._bulk(
                [{"_op_type": "delete", "_index": self.index_name, "_id": doc_id} for doc_id in ids]
            )
            deleted += len(ids)

        logger.info(f"Deleted {deleted} OpenSearch documents for {ticker.upper().strip()}")
        return deleted

    def count_documents(self, filters: Optional[SearchFilters] = None) -> int:
        """Count documents using optional exact-match metadata filters."""
        filter_query = self._to_filter_query(filters)
        try:
            response = self._client.count(
                index=self.index_name,
                body={"query": filter_query} if filter_query else None,
            )
        except NotFoundError:
            return 0
        return int(response["count"])

    def get_stats(self) -> dict[str, Any]:
        """Return backend stats in the same high-level shape as the other backends."""
        exists = bool(self._client.indices.exists(index=self.index_name))
        return {
            "backend": "opensearch",
            "collection_name": self.index_name,
            "total_documents": self.count_documents(),
            "url": self.url,
            "status": "ready" if exists else "missing",
        }

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

    def _ensure_index(self, dimension: int) -> None:
        if self._client.indices.exists(index=self.index_name):
            return
        try:
            self._client.indices.create(
                index=self.index_name,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "date_detection": False,
                        "dynamic_templates": [
                            {
                                "strings_as_keyword": {
                                    "match_mapping_type": "string",
                                    "mapping": {"type": "keyword"},
                                }
                            }
                        ],
                        "properties": {
                            "text": {"type": "text"},
                            VECTOR_FIELD: {
                                "type": "knn_vector",
                                "dimension": dimension,
                                "method": {
                                    "name": "hnsw",
                                    "engine": "lucene",
                                    "space_type": "cosinesimil",
                                },
                            },
                        },
                    },
                },
            )
        except RequestError as exc:
            if exc.error != "resource_already_exists_exception":
                raise
            return
        logger.info(f"Created OpenSearch index '{self.index_name}' with dimension={dimension}")

    def _bulk(self, actions: list[dict[str, Any]]) -> None:
        try:
            helpers.bulk(self._client, actions, refresh="wait_for")
        except helpers.BulkIndexError as exc:
            raise RuntimeError(
                f"OpenSearch bulk write failed for {len(exc.errors)} items; first: {exc.errors[0]}"
            ) from exc

    @staticmethod
    def _to_filter_query(filters: Optional[SearchFilters]) -> Optional[dict[str, Any]]:
        backend_filter = filters.to_backend_filter() if filters else None
        if backend_filter is None:
            return None

        terms: list[dict[str, Any]] = []
        for clause in backend_filter.get("$and", [backend_filter]):
            for field, value in clause.items():
                if field not in FILTER_FIELDS or not isinstance(value, (str, int, float, bool)):
                    raise ValueError(f"Invalid search filter {field!r}={value!r}")
                terms.append({"term": {field: value}})
        return {"bool": {"filter": terms}}
