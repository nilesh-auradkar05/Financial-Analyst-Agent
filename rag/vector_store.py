"""
Vector Store Module

This module provides vector storage and retrieval using ChromaDB.

Usecase in this project:
    - Store SEC filing chunks with metadata (ticker, section, date)
    - Query by semantic similarity to find relevant chunks.
    - Filter by ticker/section to narrow results

Usage:
    from rag.vector_store import ChromaDBVectorStore, get_vector_store

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

import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings
from rag.embeddings import get_embeddings
from rag.evidence import EvidencePacket

# DATA MODELS

CANONICAL_SECTION_DISPLAY_NAMES: dict[str, str] = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "md&a": "MD&A",
    "market_risk": "Market Risk",
}

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "business": (
        "business",
        "item 1",
        "item1",
        "company business",
    ),
    "risk_factors": (
        "risk factors",
        "risk factor",
        "item 1a",
        "item1a",
    ),
    "md&a": (
        "md&a",
        "management discussion and analysis",
        "managements discussion and analysis",
        "management discussion & analysis",
        "item 7",
        "item7",
    ),
    "market_risk": (
        "market risk",
        "quantitative and qualitative disclosures about market risk",
        "item 7a",
        "item7a",
    ),
}

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
        where: dict[str, Any] = {}

        if self.ticker:
            where["ticker"] = self.ticker.upper()
        if self.filing_type:
            where["filing_type"] = self.filing_type
        if self.section_key:
            where["section_key"] = self.section_key
        elif self.section_name:
            where["section"] = self.section_name
        if self.filing_date:
            where["filing_date"] = self.filing_date
        if self.extra:
            where.update(self.extra)

        return where or None

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

        ChromaDB uses L2 distance by default, where:
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
            header_parts.append(f"Field: {self.filing_date}")

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
    filter_used: Optional[dict] = None

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

        context_parts = [f"Retrieved {len(self.chunks)} relevant chunks: \n"]

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

class RetrievalStore(Protocol):
    def add_documents(self, documents: list[IndexDocument]) -> int:
        ...

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 5,
    ) -> SearchResult:
        ...

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        top_k: int = 8,
        query: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> SearchResult:
        ...

    def delete_by_ticker(self, ticker: str) -> int:
        ...

    def get_stats(self) -> dict[str, Any]:
        ...

def _normalize_section_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace("&", " and ")
    token = token.replace("’", "")
    token = token.replace("'", "")
    token = re.sub(r"[^a-z0-9]+", " ", token)
    return re.sub(r"\s+", " ", token).strip()

def canonical_section_key(section_name: Optional[str]) -> Optional[str]:
    if not section_name:
        return None

    normalized = _normalize_section_token(section_name)
    for section_key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_token = _normalize_section_token(alias)
            if normalized == alias_token or normalized.startswith(f"{alias_token} "):
                return section_key

    return None

def canonical_section_display_name(section_name: Optional[str]) -> Optional[str]:
    key = canonical_section_key(section_name)
    if key is None:
        return None

    return CANONICAL_SECTION_DISPLAY_NAMES[key]


# Vector Store Implementation
class ChromaDBVectorStore:
    """
    Vector store for SEC filings using ChromaDB.

    The class handles:
        - Collection management (create, get, delete)
        - Document indexing with embeddings
        - Semantic search with metadata filtering
        - Persistence to disk

    Example:
    ----------------------
        store = ChromaDBVectorStore()

        # Add documents
        store.add_documents(
            texts=["Risk factor text....", "Another risk...."],
            metadatas=[
                {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-01-01"},
                {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-01-02"},
            ],
        )

        # search
        results = store.search(
            query="supply chain risks",
            n_results=5,
            filter={"ticker": "AAPL"},
        )

        for chunk in results.chunks:
            print(f"{chunk.relevance_score:.2%}: {chunk.text[:100]}...")
    """
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the ChromaDBVectorStore.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (default: in-memory)
        """
        self.collection_name = collection_name or settings.chroma.collection_name
        self.persist_directory = persist_directory or settings.chroma.persist_dir

        # Initialize ChromaDB client
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
            logger.info(f"ChromaDB initialized (persistent: {self.persist_directory})")
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB initialized (in-memory)")

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "SEC Filings for Financial Analysis."
            },
        )

        self._embeddings = get_embeddings()

        logger.info(
            f"Collection '{self.collection_name}' ready "
            f"({self._collection.count()} documents)"
        )

    @property
    def collection(self):
        """Get the ChromaDB collection"""
        return self._collection

    @property
    def count(self) -> int:
        """Get number of documents in the collection"""
        return self._collection.count()

    def add_documents(
        self,
        documents: Optional[list[IndexDocument]] = None,
        *,
        texts: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> int:

        """
        Add documents to the vector store.

        Args:
            texts: List of text content to index
            metadatas: list of metadata dictionaries (one per text)
            ids: List of unique IDs (auto-generated if not provided)

        Returns:
            Numnber of documents added

        Example:
            store.add_documents(
                texts=["Risk factor 1", "Risk factor 2"],
                metadatas=[
                    {"ticker": "AAPL", "section": "Risk Factors"},
                    {"ticker": "AAPL", "section": "Risk Factors"},
                ],
            )
        """
        normalized_documents = self._normalize_documents(
            documents=documents,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        if not normalized_documents:
            return 0

        payload_ids = [doc.id for doc in normalized_documents]
        payload_texts = [doc.text for doc in normalized_documents]
        payload_metadatas = [doc.metadata for doc in normalized_documents]

        embeddings = self._embeddings.embed_documents(payload_texts)
        self._collection.add(
            ids=payload_ids,
            embeddings=embeddings,
            documents=payload_texts,
            metadatas=payload_metadatas,
        )

        logger.info(f"Added {len(normalized_documents)} documents (total: {self.count})")
        return len(normalized_documents)

    def add_document(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:

        """
        Add a single document to the vector store.

        Args:
            text: Text content to index
            metadata: Metadata dictionary
            id: Unique ID (auto-generated if not provided)

        Returns:
            ID of the added document
        """

        if id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            id = f"doc_{timestamp}"

        self.add_documents(
            documents=[IndexDocument(id=id, text=text, metadata=metadata or {})],
        )

        return id

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        n_results: int = 5,
    ) -> SearchResult:

        """
        Search for similar documents.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            filter: Metadata filter (e.g: {"ticker": "AAPL"})

        Returns:
            SearchResult with retrieved chunks

        Example:
            # search all documents
            results = store.search("revenue growth")

            # searcg with filter
            results = store.search("supply chain risks", filter={"ticker": "AAPL"})
        """
        start_time = time.time()
        backend_filter = filters.to_backend_filter() if filters else None

        logger.info(f"Searching: '{query}..' (top_k={n_results}, filter={backend_filter})")

        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Build query Kwargs
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if backend_filter:
            query_kwargs["where"] = backend_filter

        results = self._collection.query(**query_kwargs)

        chunks: list[RetrievedChunk] = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunks.append(RetrievedChunk(
                    id=doc_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {},
                    distance=results["distances"][0][i] if results.get("distances") else 0.0,
                ))

        search_time = (time.time() - start_time) * 1000
        logger.info(f"Found {len(chunks)} results in {search_time:.1f}ms")

        return SearchResult(
            query=query,
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=search_time,
            filter_used=backend_filter,
        )

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

        primary_result = self.search(
            query=query,
            filters=SearchFilters(
                ticker=ticker,
                filing_type=filing_type,
                section_key=section_key,
            ) if section_key else SearchFilters(
                ticker=ticker,
                filing_type=filing_type,
                section_name=legacy_section_name,
            ),
            n_results=n_results,
        )
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

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        n_results: int = 8,
        query: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> SearchResult:
        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        total_search_time_ms = 0.0

        for section_name in sections:
            section_query = query or f"{ticker} {section_name}"
            result, section_search_time_ms = self._search_section_with_fallback(
                ticker=ticker,
                section_name=section_name,
                query=section_query,
                n_results=n_results,
                filing_type=filing_type,
            )
            total_search_time_ms += section_search_time_ms

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

        """
        Search documents for a specific company.

        Method that builds the filter.

        Args:
            query: Search query
            ticker: Stock ticker symbol
            n_results: Maximum results
            section: Optional section filter (e.g: "Risk Factors)

        Returns:
            SearchResult with chunks from that company
        """

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

    def get_by_id(self, id: str) -> Optional[RetrievedChunk]:
        """
        Get a specific document by ID.

        Args:
            id: Document ID

        Returns:
            RetrievedChunk or None if not found
        """
        results = self._collection.get(
            ids=[id],
            include=["documents", "metadatas"],
        )

        if results.get("ids"):
            return RetrievedChunk(
                id=results["ids"][0],
                text=results["documents"][0] if results.get("documents") else "",
                metadata=results["metadatas"][0] if results.get("metadatas") else {},
                distance=0.0,
            )

        return None

    def delete_by_ticker(self, ticker: str) -> int:
        """
        Delete all documents for a specific ticker.

        Useful when re-indexing a company's filings.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Number of documents deleted
        """

        ticker = ticker.upper().strip()

        # Get all ids for this ticker
        results = self._collection.get(
            where={"ticker": ticker},
            include=[],
        )
        ids = results.get("ids") or []

        if not ids:
            return 0

        # Delete records
        self._collection.delete(ids=results["ids"])

        logger.info(f"Deleted {len(ids)} documents for {ticker}")

        return len(ids)

    def delete_collection(self):
        """Delete the collection"""
        self._client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")

    def _normalize_documents(
        self,
        *,
        documents: Optional[list[IndexDocument]],
        texts: Optional[list[str]],
        metadatas: Optional[list[dict[str, Any]]],
        ids: Optional[list[str]],
    ) -> list[IndexDocument]:
        if documents is not None:
            return documents
        if not texts:
            return []

        safe_metadatas = metadatas or [{} for _ in texts]
        if ids is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            ids = [f"doc_{timestamp}_{i}" for i in range(len(texts))]

        if len(texts) != len(safe_metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas, and ids must be of the same count.")

        return [
            IndexDocument(id=ids[i], text=texts[i], metadata=safe_metadatas[i])
            for i in range(len(texts))
        ]

    def get_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "collection_name": self.collection_name,
            "total_documents": self.count,
            "persist_directory": self.persist_directory,
        }

# Wrapper convenience functions

# Module level singleton
_default_store: Optional[ChromaDBVectorStore] = None

def get_vector_store() -> ChromaDBVectorStore:
    """
    Get the default vector store instance.

    Returns:
        ChromaDBVectorStore instance
    """
    global _default_store
    if _default_store is None:
        _default_store = ChromaDBVectorStore()

    return _default_store

def search_filings(query: str, ticker: Optional[str] = None, n_results: int = 5) -> SearchResult:
    """
    Search SEC filings. (Wrapper convenience function)

    Args:
        query: Search query
        ticker: Optional ticker filter
        n_results: Maximum results

    Returns:
        SearchResult with matching chunks
    """
    store = get_vector_store()

    if ticker:
        return store.search_by_ticker(query, ticker, n_results=n_results)
    return store.search(query, n_results=n_results)

# Testing

def _main():
    """Test the vector store"""

    print(f"\n{'='*60}")
    print("Vector Store - Testing")
    print(f"{'='*60}\n")

    # Create vector store
    print("1. Creating vector store....")
    store = ChromaDBVectorStore()
    print(f"    Collection: {store.collection_name}")
    print(f"    Documents: {store.count}\n")

    # Add sample documents
    print("2. Adding Sample documents....")

    sample_texts = [
        "The company faces significant supply chain risks due to concentration of manufacturing in Asia.",
        "Revenue grew 25% year-over-year driven by strong iPhone sales in emerging markets.",
        "We are subject to risks associated with changes in tax laws and regulations globally.",
        "Competition in the smartphone market has intensified with new entrants from China.",
        "Currency fluctuations may materially impact our international revenue and profitability.",
    ]

    sample_metadatas = [
        {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-10-31"},
        {"ticker": "AAPL", "section": "MD&A", "filing_date": "2024-10-31"},
        {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-10-31"},
        {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-10-31"},
        {"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2024-10-31"},
    ]

    count = store.add_documents(
        texts=sample_texts,
        metadatas=sample_metadatas,
    )
    print(f"    Added {count} documents\n")

    # Test Search
    print("3. Testing search....")

    queries = [
        ("supply chain", None),
        ("revenue growth", None),
        ("tax risks", {"section": "Risk Factors"}),
    ]

    for query, filter_dict in queries:
        results = store.search(query, n_results=3, filters=SearchFilters(**filter_dict))
        print(f"\n    Query: '{query}' (filter: {filter_dict})")
        print(f"     Found: {results.total_results} results in {results.search_time_ms:.1f}ms")

        for chunk in results.chunks[:2]:
            print(f"     - [{chunk.relevance_score:.2%}] {chunk.text[:60]}...")

    # Test ticker search
    print("\n4. Testing ticker search....")
    results = store.search_by_ticker("risks", "AAPL", n_results=3)
    print("     Query: 'risks' for AAPL")
    print(f"    Found: {results.total_results} results")

    # stats
    print("\n5. Testing Ticker search....")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print(f"{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    _main()
