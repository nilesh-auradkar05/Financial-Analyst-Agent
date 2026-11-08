"""
OpenSearch retrieval backend.

Implements the RetrievalStore contract on an OpenSearch k-NN index. Chunk IDs are the
document `_id`, so higher layers see the same IDs they wrote. Only index create/exists,
`_bulk`, `_search` and `_count` are used, so the adapter also fits OpenSearch Serverless.
The vector field names neither an engine nor a mode (Serverless rejects both) and sets
`compression_level`; searches rescore at full precision, so distances are exact at either level.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from typing import Any, Optional

import botocore.session
from langchain_core.embeddings import Embeddings
from langsmith import get_current_run_tree
from loguru import logger
from opensearchpy import (
    NotFoundError,
    OpenSearch,
    OpenSearchException,
    RequestError,
    Urllib3AWSV4SignerAuth,
    helpers,
)

from app.components.retrieval.embeddings import (
    EmbeddingDimensionMismatchError,
    embedding_identifier,
    embedding_index_name,
    get_embeddings,
)
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
SIGV4_DEFAULT_TIMEOUT_S = 30.0
COMPRESSION_LEVELS = ("1x", "32x")
DEFAULT_COMPRESSION_LEVEL = "32x"
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
ITER_PAGE_SIZE = 500
DEFAULT_AWS_REGION = "us-east-1"
SIGV4_SERVICE = "aoss"


class IndexCompressionMismatchError(ValueError):
    """The index's vector compression level differs from OPENSEARCH_COMPRESSION_LEVEL."""


def _compression_level() -> str:
    level = os.getenv("OPENSEARCH_COMPRESSION_LEVEL") or DEFAULT_COMPRESSION_LEVEL
    if level not in COMPRESSION_LEVELS:
        raise ValueError(f"OPENSEARCH_COMPRESSION_LEVEL must be '1x' or '32x', got {level!r}")
    return level


def _request_signer() -> Urllib3AWSV4SignerAuth | None:
    """Return the SigV4 signer selected by OPENSEARCH_AUTH, or None for unsigned requests."""
    mode = os.getenv("OPENSEARCH_AUTH") or "none"
    if mode == "none":
        return None
    if mode != "sigv4":
        raise ValueError(f"OPENSEARCH_AUTH must be 'none' or 'sigv4', got {mode!r}")
    credentials = botocore.session.get_session().get_credentials()
    if credentials is None:
        raise ValueError("OPENSEARCH_AUTH=sigv4 but no AWS credentials could be resolved")
    # botocore.auth logs the canonical request at DEBUG, which carries the session token.
    logging.getLogger("botocore.auth").setLevel(logging.WARNING)
    return Urllib3AWSV4SignerAuth(
        credentials, os.getenv("AWS_REGION") or DEFAULT_AWS_REGION, SIGV4_SERVICE
    )


class OpenSearchVectorStore(RetrievalStore):
    """OpenSearch-backed implementation of the retrieval contract.

    Vectors use cosine similarity on an HNSW index at the configured compression level;
    metadata filters apply inside the k-NN query, so a filtered search returns up to
    `n_results` matching chunks, rescored at full precision.
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        default_index_name = embedding_index_name(settings.chroma.collection_name or "sec_filings")
        self.index_name: str = (
            index_name or os.getenv("OPENSEARCH_INDEX_NAME") or default_index_name
        )
        self.url: str = url or os.getenv("OPENSEARCH_URL") or DEFAULT_OPENSEARCH_URL
        self._compression_level = _compression_level()
        signer = _request_signer()
        default_timeout = DEFAULT_TIMEOUT_S if signer is None else SIGV4_DEFAULT_TIMEOUT_S
        timeout = timeout_s or float(os.getenv("OPENSEARCH_TIMEOUT_S") or default_timeout)
        # Serverless rejects refresh=wait_for; its writes become searchable after a short lag.
        self._write_options: dict[str, Any] = {} if signer else {"refresh": "wait_for"}
        self._embeddings = embeddings or get_embeddings()
        self._index_verified = False
        self._client = OpenSearch(
            hosts=[self.url],
            http_auth=signer,
            timeout=timeout,
            max_retries=0 if signer is None else 1,
            retry_on_timeout=False,
        )
        logger.info(f"OpenSearch initialized (index={self.index_name})")

    @property
    def count(self) -> int:
        return self.count_documents()

    def healthcheck(self) -> bool:
        # Serverless answers `/` with 404, so probe the index; a missing index is still reachable.
        try:
            self._client.indices.exists(index=self.index_name)
        except OpenSearchException:
            return False
        return True

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

        self._ensure_index()
        vectors = self._embeddings.embed_documents([doc.text for doc in normalized_documents])

        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    **doc.metadata,
                    "text": doc.text,
                    "embedding_id": embedding_identifier(),
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
                    embedding_model=settings.embedding.model,
                    embedding_provider=settings.embedding.provider,
                )
            self._require_compatible_index()
            knn: dict[str, Any] = {
                "vector": self._embeddings.embed_query(query),
                "k": n_results,
                "rescore": True,
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

    def iter_documents(self) -> Iterator[IndexDocument]:
        """Yield every chunk (id, text, metadata) without its vector, page by page."""
        search_after: Optional[list[Any]] = None
        while True:
            body: dict[str, Any] = {
                "size": ITER_PAGE_SIZE,
                "_source": {"excludes": [VECTOR_FIELD]},
                "query": {"match_all": {}},
                "sort": [{"_id": "asc"}],
            }
            if search_after:
                body["search_after"] = search_after
            try:
                hits = self._client.search(index=self.index_name, body=body)["hits"]["hits"]
            except NotFoundError:
                return
            except OpenSearchException as exc:
                raise RuntimeError(f"OpenSearch scan failed: {type(exc).__name__}: {exc}") from exc
            for hit in hits:
                source = dict(hit["_source"])
                text = str(source.pop("text", ""))
                yield IndexDocument(id=hit["_id"], text=text, metadata=source)
            if len(hits) < ITER_PAGE_SIZE:
                return
            search_after = hits[-1]["sort"]

    def delete_by_ticker(self, ticker: str) -> int:
        """Delete all documents for a ticker."""
        filter_query = self._to_filter_query(SearchFilters(ticker=ticker.upper().strip()))
        ids: list[str] = []
        search_after: Optional[list[Any]] = None

        # Page every id first and never search again after deleting: Serverless search is
        # eventually consistent, so deleted ids would keep coming back.
        while True:
            body: dict[str, Any] = {
                "size": DELETE_PAGE_SIZE,
                "_source": False,
                "query": filter_query,
                "sort": [{"_id": "asc"}],
            }
            if search_after:
                body["search_after"] = search_after
            try:
                hits = self._client.search(index=self.index_name, body=body)["hits"]["hits"]
            except NotFoundError:
                break
            ids.extend(hit["_id"] for hit in hits)
            if len(hits) < DELETE_PAGE_SIZE:
                break
            search_after = hits[-1]["sort"]

        self._bulk(
            [{"_op_type": "delete", "_index": self.index_name, "_id": doc_id} for doc_id in ids]
        )
        logger.info(f"Deleted {len(ids)} OpenSearch documents for {ticker.upper().strip()}")
        return len(ids)

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

    def _require_compatible_index(self) -> None:
        """Refuse an existing index whose vector size or compression differs from the configuration."""
        if self._index_verified:
            return
        mapping = self._client.indices.get_mapping(index=self.index_name)
        vector = mapping[self.index_name]["mappings"]["properties"][VECTOR_FIELD]
        found = int(vector["dimension"])
        configured = settings.embedding.dimensions
        if found != configured:
            raise EmbeddingDimensionMismatchError(
                f"Index {self.index_name!r} holds {found}-dimensional vectors but "
                f"EMBEDDING_DIMENSIONS is {configured}"
            )
        found_level = vector.get("compression_level", "1x")
        if found_level != self._compression_level:
            raise IndexCompressionMismatchError(
                f"Index {self.index_name!r} uses compression level {found_level!r} but "
                f"OPENSEARCH_COMPRESSION_LEVEL is {self._compression_level!r}"
            )
        self._index_verified = True

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self.index_name):
            self._require_compatible_index()
            return
        dimension = settings.embedding.dimensions
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
                                "compression_level": self._compression_level,
                                "method": {"name": "hnsw", "space_type": "cosinesimil"},
                            },
                        },
                    },
                },
            )
        except RequestError as exc:
            if exc.error != "resource_already_exists_exception":
                raise
            return
        self._index_verified = True
        logger.info(f"Created OpenSearch index '{self.index_name}' with dimension={dimension}")

    def _bulk(self, actions: list[dict[str, Any]]) -> None:
        _, errors = helpers.bulk(
            self._client, actions, raise_on_error=False, **self._write_options
        )
        # A delete of an id that is already gone is the outcome the caller asked for.
        failures = [
            item
            for item in errors
            if not any(op == "delete" and detail.get("status") == 404 for op, detail in item.items())
        ]
        if failures:
            raise RuntimeError(
                f"OpenSearch bulk write failed for {len(failures)} items; first: {failures[0]}"
            )

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
