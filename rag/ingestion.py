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
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import get_vector_store
from tools.sec_filings_tool import Filing, get_latest_10k

# Data Models

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
    filing_date: Optional[str] = None
    total_chunks: int = 0
    sections_processed: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.total_chunks > 0

# Text Splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
    length_function=len,
)

# Default sections to extract from 10-K filings
DEFAULT_SECTIONS = [
    "Business",
    "Risk Factors",
    "MD&A",
    "Market Risk",
]

# Ingestion functions

@traceable(name="ingest_filing", run_type="chain", tags=["rag", "ingestion"])
def ingest_filing(
    filing: Filing,
    sections: list[str] | None = None,
    replace_existing: bool = False,
) -> IngestionResult:
    """
    Ingest a filing into the vector store.
    
    Args:
        filing: Filing object with sections
        sections: Sections to process (defaults to DEFAULT_SECTIONS)
        replace_existing: Delete existing chunks for this ticker first
        
    Returns:
        IngestionResult with status
    """
    start = datetime.now(timezone.utc)
    
    ticker: str = filing.metadata.ticker or ""
    filing_type = filing.metadata.filing_type
    
    logger.info(f"Ingesting {filing_type} for {ticker}")
    
    store = get_vector_store()
    
    if not replace_existing:
        existing = store._collection.get(
            where={"ticker": ticker.upper()},
            include=[],
        )
        existing_count = len(existing.get("ids", []))
        if existing_count:
            logger.info(f"Chunks already present for {ticker} ({existing_count})")
            return IngestionResult(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing.metadata.filing_date,
                total_chunks=existing_count,
                sections_processed=[],
            )

    # Delete existing if requested
    if replace_existing:
        deleted = store.delete_by_ticker(ticker)
        if deleted:
            logger.info(f"Deleted {deleted} existing chunks")
    
    # Process sections
    sections_to_process = sections or DEFAULT_SECTIONS
    all_texts = []
    all_metadatas = []
    all_ids = []
    processed_sections = []
    
    for section_name in sections_to_process:
        if section_name not in filing.sections:
            continue
        
        section = filing.sections[section_name]
        
        # Use LangChain's splitter
        chunks = text_splitter.split_text(section.content)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{ticker}_{filing_type}_{section_name.replace(' ', '_')}_{i:03d}"
            
            all_texts.append(chunk)
            all_metadatas.append({
                "ticker": ticker.upper(),
                "section": section_name,
                "filing_type": filing_type,
                "filing_date": filing.metadata.filing_date,
                "chunk_index": i,
            })
            all_ids.append(chunk_id)
        
        processed_sections.append(section_name)
        logger.info(f"  {section_name}: {len(chunks)} chunks")
    
    if not all_texts:
        return IngestionResult(
            ticker=ticker,
            filing_type=filing_type,
            error="No content found in sections",
        )
    
    # Add to vector store
    store.add_documents(all_texts, all_metadatas, all_ids)

    duration = datetime.now(timezone.utc) - start
    logger.info(f"Ingested {len(all_texts)} chunks in {duration.total_seconds():.1f}s")
    
    return IngestionResult(
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing.metadata.filing_date,
        total_chunks=len(all_texts),
        sections_processed=processed_sections,
    )

@traceable(name="ingest_10k_for_ticker", run_type="chain", tags=["rag", "10k"])
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
    logger.info(f"Starting 10-K ingestion for {ticker}")
    
    # Download filing
    filing = await get_latest_10k(ticker)
    
    if not filing:
        return IngestionResult(
            ticker=ticker,
            filing_type="10-K",
            error="Filing not found",
        )
    
    if not filing.success:
        return IngestionResult(
            ticker=ticker,
            filing_type="10-K",
            error=filing.error or "Failed to download filing",
        )
    
    # Ingest
    return ingest_filing(filing)

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
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"\nIngesting 10-K for {ticker}...\n")
    
    result = await ingest_10k_for_ticker(ticker)
    
    if result.success:
        print(" Success!")
        print(f"   Chunks: {result.total_chunks}")
        print(f"   Sections: {', '.join(result.sections_processed)}")
        print(f"   Filing date: {result.filing_date}")
    else:
        print(f"Failed to ingest 10-K for {ticker}: {result.error}")
    
if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    asyncio.run(_main())