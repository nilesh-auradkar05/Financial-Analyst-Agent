"""
Vector Store Module

This module defines the vector-store types and the factory for the OpenSearch store.

Usecase in this project:
    - Store SEC filing chunks with metadata (ticker, section, date)
    - Query by semantic similarity to find relevant chunks.
    - Filter by ticker/section to narrow results

Usage:
    from app.components.retrieval.vector_store import get_vector_store

    # Get/create vector store
    store = get_vector_store()

    # Add documents
    store.add_documents(
        texts=["Risk factor 1...", "Risk factor 2..."],
        metadatas=[{"ticker": "AAPL", "section": "Risk Factors"}, ....],
        ids=["AAPL_risk_001", "AAPL_risk_002"],
    )

    # Search
    results = store.search("supply chain risks", filter={"ticker": "AAPL"})
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, cast

from app.components.retrieval.evidence import EvidencePacket
from app.components.retrieval.sections import (  # noqa: F401  re-exported for importers
    canonical_section_display_name,
    canonical_section_key,
)

# DATA MODELS

@dataclass(frozen=True, slots=True)
class IndexDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SearchFilters:
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    section_name: Optional[str] = None
    section_key: Optional[str] = None
    filing_date: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_backend_filter(self) -> Optional[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []

        if self.ticker:
            clauses.append({"ticker": self.ticker.upper()})
        if self.filing_type:
            clauses.append({"filing_type": self.filing_type})
        if self.section_key:
            clauses.append({"section_key": self.section_key})
        elif self.section_name:
            clauses.append({"section": self.section_name})
        if self.filing_date:
            clauses.append({"filing_date": self.filing_date})
        if self.extra:
            for k, v in self.extra.items():
                clauses.append({k: v})

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

@dataclass
class RetrievedChunk:
    """
    A single chunk retrieved from vector search.

    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        metadata: Associated metadata (ticker, section, date, etc.)
        distance: Distance from query (lower = more similar)
        relevance_score: Converted to 0-1 relevance (higher = more relevant)
    """

    id: str
    text: str
    metadata: dict[str, Any] | Any
    distance: float

    @property
    def relevance_score(self) -> float:
        """
        Convert distance to relevance score (0-1)

        Distance is L2-style, where:
            - 0 = identical
            - Higher = less similar

        We convert to relevance where 1 = most relevant.
        """
        return 1 / (1 + self.distance)

    @property
    def ticker(self) -> Optional[str]:
        """Get ticker from metadata"""
        return self.metadata.get("ticker")

    @property
    def section(self) -> Optional[str]:
        """Get section name from metadata"""
        return self.metadata.get("section_name") or self.metadata.get("section")

    @property
    def filing_date(self) -> Optional[str]:
        """Get filing date from metadata"""
        return self.metadata.get("filing_date")

    def to_context(self) -> str:
        """Format chunk as context for LLM."""
        header_parts = []
        if self.ticker:
            header_parts.append(f"Ticker: {self.ticker}")
        if self.section:
            header_parts.append(f"Section: {self.section}")
        if self.filing_date:
            header_parts.append(f"Filing: {self.filing_date}")

        header = " | ".join(header_parts) if header_parts else "Document Chunk"

        return f"[{header}]\n{self.text}"

@dataclass
class SearchResult:
    """
    Result of a vector search operation.

    Attributes:
        query: The search query
        chunks: List of retrieved chunks
        total_results: Number of results returned
        search_time_ms: Time taken for search
        filter_used: Metadata filter that was applied
    """

    query: str
    chunks: list[RetrievedChunk]
    total_results: int = 0
    search_time_ms: float = 0.0
    filter_used: Optional[dict[str, Any]] = None

    @property
    def has_results(self) -> bool:
        """Return if search returned any results"""
        return len(self.chunks) > 0

    def to_context(self, max_chunks: int = 5) -> str:
        """Format top chunks as context for LLM.

        Args:
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string
        """
        if not self.chunks:
            return "No relevant documents found."

        context_parts = [f"Retrieved {len(self.chunks)} relevant chunks:\n"]

        for i, chunk in enumerate(self.chunks[:max_chunks], start=1):
            context_parts.append(f"--- Chunk {i} (relevance: {chunk.relevance_score:.2%}) ---")
            context_parts.append(chunk.to_context())
            context_parts.append("")

        return "\n".join(context_parts)

    def to_evidence_packets(self, retrieval_method: str = "dense") -> list[EvidencePacket]:
        return [
            EvidencePacket.from_chunk(
                chunk,
                retrieval_method=retrieval_method,
                rank=rank,
            )
            for rank, chunk in enumerate(self.chunks, start=1)
        ]

VectorBackend = Literal["opensearch"]

class RetrievalStore(Protocol):
    @property
    def count(self) -> int:
        ...

    def add_documents(self, documents: list[IndexDocument]) -> int:
        ...

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:
        ...

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        n_results: int = 8,
        query: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> SearchResult:
        ...

    def search_by_ticker(
        self,
        query: str,
        ticker: str,
        n_results: int = 5,
        section: Optional[str] = None,
    ) -> SearchResult:
        ...

    def delete_by_ticker(self, ticker: str) -> int:
        ...

    def count_documents(self, filters: Optional[SearchFilters] = None) -> int:
        ...

    def get_stats(self) -> dict[str, Any]:
        ...

def _sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        sanitized = _sanitize_metadata_value(value)
        if sanitized is not None:
            cleaned[key] = sanitized
    return cleaned


# Wrapper convenience functions

# Module level singleton
_default_store: Optional[RetrievalStore] = None
_default_backend: Optional[VectorBackend] = None

def _get_vector_backend() -> VectorBackend:
    backend = os.getenv("VECTOR_BACKEND", "opensearch").strip().lower()
    if backend != "opensearch":
        raise ValueError(
            f"Unsupported VECTOR_BACKEND: {backend!r}. The only accepted backend is 'opensearch'."
        )
    return cast(VectorBackend, backend)

def reset_vector_store() -> None:
    """Reset the process-local vector store singleton

    For testing only.
    """
    global _default_store, _default_backend
    _default_store = None
    _default_backend = None


def get_vector_store() -> RetrievalStore:
    """
    Get the configured vector store instance.

    VECTOR_BACKEND defaults to `opensearch`, the only accepted value.
    Returns:
        RetrievalStore instance
    """

    global _default_store, _default_backend

    backend = _get_vector_backend()
    if _default_store is not None and _default_backend == backend:
        return _default_store

    from app.components.retrieval.opensearch_store import OpenSearchVectorStore

    _default_store = OpenSearchVectorStore()

    _default_backend = backend
    return _default_store
