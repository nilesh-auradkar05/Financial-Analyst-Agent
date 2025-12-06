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

import chromadb
from chromadb.config import Settings as ChromaSettings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings
from rag.embeddings import get_embeddings

# DATA MODELS

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
    metadata: dict[str, Any]
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
        return self.metadata.get("section")

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

        for i, chunk in enumerate(self.chunks[:max_chunks], 1):
            context_parts.append(f"--- Chunk {i} (relevance: {chunk.relevance_score:.2%}) ---")
            context_parts.append(chunk.to_context())
            context_parts.append("")

        return "\n".join(context_parts)

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
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
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
        if not texts:
            return 0

        # Generate IDs if not provided
        if ids is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            ids = [f"doc_{timestamp}_{i}" for i in range(len(texts))]

        # Default empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Validate lengths match
        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas, and ids must have same length")

        logger.info(f"Adding {len(texts)} documents to collection")

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(texts)

        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(texts)} documents (total: {self.count})")

        return len(texts)

    def add_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
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
            texts=[text],
            metadatas=[metadata or {}],
            ids=[id],
        )

        return id

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter: Optional[dict] = None,
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
        import time
        start_time = time.time()

        logger.info(f"Searching: '{query}..' (n={n_results}, filter={filter})")

        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Build query Kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter:
            query_kwargs["where"] = filter

        results = self._collection.query(**query_kwargs)

        chunks = []

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunks.append(RetrievedChunk(
                    id=doc_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"][0][i] else {},
                    distance=results["distances"][0][i] if results["distances"] else 0.0,
                ))

        search_time = (time.time() - start_time) * 1000

        logger.info(f"Found {len(chunks)} results in {search_time:.1f}ms")

        return SearchResult(
            query=query,
            chunks=chunks,
            total_results=len(chunks),
            search_time_ms=search_time,
            filter_used=filter,
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
        filter_dict = {"ticker": ticker.upper()}

        if section:
            filter_dict["section"] = section
        return self.search(query, n_results=n_results, filter=filter_dict)

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

        if results["ids"]:
            return RetrievedChunk(
                id=results["ids"][0],
                text=results["documents"][0] if results["documents"][0] else "",
                metada=results["metadatas"][0] if results["metadatas"][0] else {},
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

        if not results["ids"]:
            return 0

        count = len(results["ids"])

        # Delete records
        self._collection.delete(ids=results["ids"])

        logger.info(f"Deleted {count} documents for {ticker}")

        return count

    def delete_collection(self):
        """Delete the collection"""
        self._collection.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")

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
        return store.search_by_ticker(query, ticker, n_results)
    else:
        return store.search(query, n_results)

# Testing

def _main():
    """Test the vector store"""
    import sys

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
        results = store.search(query, n_results=3, filter=filter_dict)
        print(f"\n    Query: '{query}' (filter: {filter_dict})")
        print(f"     Found: {results.total_results} results in {results.search_time_ms:.1f}ms")

        for chunk in results.chunks[:2]:
            print(f"     - [{chunk.relevance_score:.2%}] {chunk.text[:60]}...")

    # Test ticker search
    print("\n4. Testing ticker search....")
    results = store.search_by_ticker("risks", "AAPL", n_results=3)
    print(f"    Query: 'risks' for AAPL")
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