"""
Document Ingestion Module

This module handles the ingeestion of SEC filings into the vector store.

Ingestion Pipeline:
    1. Load filing (from disk or Download from EDGAR)
    2. Extract sections
    3. Chunk sections into smaller chunks
    4. Generate embeddings
    5. Store in ChromDB with metadata

Usage:
    from rag.ingestion import ingest_filing, ingest_10k_for_ticker
    
    # Ingest a filing object
    from tools.sec_filings import download_10k
    filing = await download_10k("AAPL")
    count = ingest_filing(filing)
    
    # Or ingest directly by ticker
    count = await ingest_10k_for_ticker("AAPL")
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from loguru import logger
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings
from rag.vector_store import ChromaDBVectorStore, get_vector_store
from tools.sec_filings_tool import Filing, FilingSection, SECFilingsTool

# DATA MODELS

@dataclass
class Chunk:
    """
    A single chunk of text ready for indexing.
    
    Attributes:
        id: Unique identifier
        text: Chunk text content
        metadata: Associated metadata
        char_count: Number of characters
        word_count: Number of words
    """

    id: str
    text: str
    metadata: dict

    @property
    def char_count(self) -> int:
        """Number of characters in chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Number of words in chunk."""
        return len(self.text.split())

@dataclass
class IngestionResult:
    """
    Result of ingestion operation.
    
    Attributes:
        ticker: Stock ticker
        filing_type: Type of filing (10-K, 10-Q)
        filing_date: Date of filing
        total_chunks: Number of chunks created
        sections_processed: List of section names processed
        total_characters: Total characters processed
        error: Error message if ingestion failed
    """

    ticker: str
    filing_type: str
    filing_date: str
    total_chunks: int = 0
    sections_processed: list[str] = field(default_factory=list)
    total_characters: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.total_chunks > 0

# Text Chunker

class TextChunker:
    """
    Splits text into chunks for vector indexing.
    
    Uses recursive splitting strategy:
    1. Try to split on double newlines (paragraph boundaries)
    2. Fall back to single newlines
    3. Fall back to sentence boundaries
    4. Fall back to spaces
    
    Example:
    --------
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.split_text(long_text)
    """
    SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        ", ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):

        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Overlap between chunks (characters)
            separators: Custom separator list (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.SEPARATORS

        logger.debug(f"TextChunker: size={chunk_size}, overlap={chunk_overlap}")

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split recursively
        chunks = self._split_recursive(text, self.separators)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace."""
        # Replace multiple spaces/newlines with single
        text = re.sub(r'\s+', ' ', text)
        
        # But preserve paragraph breaks
        text = re.sub(r' *\n *', '\n\n', text)
        return text.strip()

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:

        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
            
        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators, split by size
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator = split every character
            splits = list(text)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back (except for character split)
            piece = split + separator if separator else split
            
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # Current chunk is full
                if current_chunk:
                    # Recursively split if still too large
                    if len(current_chunk) > self.chunk_size:
                        chunks.extend(
                            self._split_recursive(current_chunk, remaining_separators)
                        )
                    else:
                        chunks.append(current_chunk.strip())
                
                current_chunk = piece
        
        # Don't forget the last chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                chunks.extend(
                    self._split_recursive(current_chunk, remaining_separators)
                )
            else:
                chunks.append(current_chunk.strip())
        
        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        """Split text by size when no separator works."""
        chunks = []

        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i+self.chunk_size])

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between chunks for context continuity"""

        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:]

            # Find a clean break point (word boundary)
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]

            # Prepend overlap to current chunk
            overlapped.append(overlap_text + curr_chunk)

        return overlapped

# DOCUMENT INGESTION

class FilingIngester:
    """
    Ingests SEC filings into the vector store.
    
    Handles the full pipeline:
    1. Extract relevant sections from filing
    2. Chunk each section
    3. Generate IDs and metadata
    4. Store in vector database
    
    Example:
    --------
        ingester = FilingIngester()
        
        # Ingest a filing
        result = ingester.ingest(filing)
        print(f"Created {result.total_chunks} chunks")
        
        # Ingest specific sections only
        result = ingester.ingest(filing, sections=["Risk Factors", "MD&A"])
    """

    DEFAULT_SECTIONS = [
        "Business",
        "Risk Factors",
        "Management's Discussion and Analysis",
        "Quantitative and Qualitative Disclosures About Market Risk",
    ]

    def __init__(
        self,
        vector_store: Optional[ChromaDBVectorStore] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the ingester.
        
        Args:
            vector_store: VectorStore instance (uses default if not provided)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store or get_vector_store()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info(
            "FilingIngester initialized "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )

    def ingest(
        self,
        filing: Filing,
        sections: Optional[list[str]] = None,
        replace_existing: bool = True,
    ) -> IngestionResult:
        """
        Ingest a filing into the vector store.
        
        Args:
            filing: Filing object to ingest
            sections: List of sections to ingest (default: all key sections)
            replace_existing: Delete existing chunks for this ticker first
            
        Returns:
            IngestionResult with statistics
        """
        ticker = filing.ticker
        filing_type = filing.filing_type
        filing_date = filing.filing_date

        logger.info(f"Ingesting {filing_type} for {ticker} (filed: {filing_date})")

        # Delete existing if requested
        if replace_existing:
            deleted = self.vector_store.delete_by_ticker(ticker)
            if deleted > 0:
                logger.info(f"Deleted {deleted} existing chunks for {ticker}")

        # Determine which sections to process
        sections_to_process = sections or self.DEFAULT_SECTIONS

        all_chunks = []
        processed_sections = []
        total_chars = 0

        for section_name in sections_to_process:
            section = filing.get_section(section_name)

            if not section:
                logger.debug(f"Section not found: {section_name}")
                continue

            # Chunk the section
            chunks = self._chunk_section(
                section=section,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
            )

            all_chunks.extend(chunks)
            processed_sections.append(section_name)
            total_chars += section.word_count

        if not all_chunks:
            return IngestionResult(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                error="No sections found to ingest",
            )

        # Add to vector store
        texts = [chunk.text for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [chunk.id for chunk in all_chunks]

        self.vector_store.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Ingested {len(all_chunks)} chunks from {len(processed_sections)} sections")

        return IngestionResult(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            total_chunks=len(all_chunks),
            sections_processed=processed_sections,
            total_characters=total_chars,
        )

    def _chunk_section(
        self,
        section: FilingSection,
        ticker: str,
        filing_type: str,
        filing_date: str,
    ) -> list[Chunk]:
        """
        Chunk a single section.
        
        Args:
            section: FilingSection to chunk
            ticker: Stock ticker
            filing_type: Filing type
            filing_date: Filing date
            
        Returns:
            List of Chunk objects
        """
        text_chunks = self.chunker.split_text(section.content)

        chunks = []

        for i, text in enumerate(text_chunks):
            # Create Unique ID
            chunk_id = f"{ticker}_{filing_type}_{filing_date}_{section.name}_{i:04d}"
            chunk_id = chunk_id.replace(" ", "_").replace("/", "-")

            # Build metadata (filter out None values - ChromaDB doesn't accept them)
            metadata = {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section": section.name,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            }
            # Only add item_number if it's not None
            if section.item_number is not None:
                metadata["item_number"] = section.item_number

            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
            ))

        logger.debug(
            f"Section '{section.name}': {len(text_chunks)} chunks "
            f"({section.word_count} words)"
        )

        return chunks

    def ingest_from_text(
        self,
        text: str,
        ticker: str,
        section: str = "Custom",
        filing_type: str = "10-K",
        filing_date: Optional[str] = None,
    ) -> IngestionResult:

        """
        Ingest raw text (useful for testing or custom documents).
        
        Args:
            text: Text to ingest
            ticker: Stock ticker
            section: Section name
            filing_type: Filing type
            filing_date: Filing date
            
        Returns:
            IngestionResult
        """
        filing_date = filing_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Create a FilingSection
        section_obj = FilingSection(
            name=section,
            content=text,
        )

        # Chunk it
        chunks = self._chunk_section(
            section=section_obj,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
        )

        if not chunks:
            return IngestionResult(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                error="No chunks created from text",
            )

        # Add to vector store
        self.vector_store.add_documents(
            texts=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
        )

        return IngestionResult(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            total_chunks=len(chunks),
            sections_processed=[section],
            total_characters=len(text),
        )

# WRAPPER Convenience functions

def ingest_filing(
    filing: Filing,
    sections: Optional[list[str]] = None,
) -> IngestionResult:
    """
    Ingest a filing into the vector store (convenience function).
    
    Args:
        filing: Filing object
        sections: Sections to ingest (optional)
        
    Returns:
        IngestionResult
    """
    ingester = FilingIngester()
    return ingester.ingest(filing, sections=sections)

async def ingest_10k_for_ticker(
    ticker: str,
    sections: Optional[list[str]] = None,
) -> IngestionResult:

    """
    Download and ingest the latest 10-K for a ticker.
    
    This is the main convenience function for indexing a company.
    
    Args:
        ticker: Stock ticker symbol
        sections: Sections to ingest (optional)
        
    Returns:
        IngestionResult
        
    Example:
        result = await ingest_10k_for_ticker("AAPL")
        print(f"Ingested {result.total_chunks} chunks")
    """
    ticker = ticker.upper()

    logger.info(f"Downloading and ingest 10-K for Pticker")

    async with SECFilingsTool() as tool:
        filing = await tool.get_latest_10k(ticker)

    if not filing:
        return IngestionResult(
            ticker=ticker,
            filing_type="10-K",
            filing_date="",
            error=f"Could not download 10-K for {ticker}",
        )

    # Ingest doc
    ingester = FilingIngester()
    return ingester.ingest(filing, sections=sections)

def get_ingestion_stats() -> dict:
    """
    Get statistics about the ingested documents.

    Returns:
        Dict with ingestion statistics.
    """
    store = get_vector_store()
    return store.get_stats()

# Testing
async def _main():
    """Test the ingestion pipeline."""
    import sys
    
    print(f"\n{'='*60}")
    print("Document Ingestion - Testing")
    print(f"{'='*60}\n")
    
    # Get ticker from args or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    # Test chunker
    print("1. Testing Text Chunker...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
    The company faces significant risks related to its supply chain operations. 
    Manufacturing is concentrated in Asia, particularly China, which exposes 
    the company to geopolitical tensions, trade disputes, and regulatory changes.
    
    Additionally, the company relies on a limited number of suppliers for critical 
    components. Any disruption to these suppliers could materially impact production 
    and revenue.
    
    Currency fluctuations also pose a risk to international operations. The company 
    generates approximately 60% of revenue from international markets, making it 
    vulnerable to exchange rate movements.
    
    Competition in the technology sector continues to intensify. New entrants and 
    established competitors are investing heavily in research and development, 
    potentially eroding market share and margins.
    """ * 3  # Make it longer
    
    chunks = chunker.split_text(sample_text)
    print(f"   Split {len(sample_text)} chars into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   Chunk {i+1}: {len(chunk)} chars")
    print()
    
    # Test text ingestion
    print("2. Testing Text Ingestion...")
    ingester = FilingIngester(chunk_size=500, chunk_overlap=100)
    
    result = ingester.ingest_from_text(
        text=sample_text,
        ticker="TEST",
        section="Risk Factors",
    )
    
    if result.success:
        print(f"   Ingested {result.total_chunks} chunks")
    else:
        print(f"   Failed: {result.error}")
    print()
    
    # Test search
    print("3. Testing Search on Ingested Content...")
    from rag.vector_store import get_vector_store
    
    store = get_vector_store()
    results = store.search("supply chain risks", n_results=3)
    
    print(f"   Query: 'supply chain risks'")
    print(f"   Found: {results.total_results} results")
    for chunk in results.chunks[:2]:
        print(f"   - [{chunk.relevance_score:.2%}] {chunk.text[:80]}...")
    print()
    
    # Test full ingestion (if ticker provided)
    print(f"4. Testing Full 10-K Ingestion for {ticker}...")
    print("   (This may take a minute to download and process...)")
    
    result = await ingest_10k_for_ticker(ticker)
    
    if result.success:
        print(f"   Ingested {result.total_chunks} chunks")
        print(f"   Sections: {', '.join(result.sections_processed)}")
        print(f"   Filing Date: {result.filing_date}")
    else:
        print(f"   Failed: {result.error}")
    
    # Final stats
    print("\n5. Final Statistics...")
    stats = get_ingestion_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n{'='*60}")
    print("All tests complete!")
    print(f"{'='*60}")
    
if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    asyncio.run(_main())