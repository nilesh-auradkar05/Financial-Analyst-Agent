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

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import IndexDocument, get_vector_store
from tools.sec_filings_tool import Filing, get_latest_10k

# Data Models

@dataclass(frozen=True, slots=True)
class CanonicalSection:
    key: str
    display_name: str
    slug: str

@dataclass(slots=True)
class BuildDocumentsResult:
    documents: list[IndexDocument]
    sections_requested: list[str] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    sections_skipped: list[str] = field(default_factory=list)

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
    sections_requested: list[str] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    sections_skipped: list[str] = field(default_factory=list)
    documents_written: int = 0

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

CANONICAL_SECTIONS: dict[str, CanonicalSection] = {
    "business": CanonicalSection(
        key="business",
        display_name="Business",
        slug="business",
    ),
    "risk_factors": CanonicalSection(
        key="risk_factors",
        display_name="Risk Factors",
        slug="risk_factors",
    ),
    "md&a": CanonicalSection(
        key="md&a",
        display_name="MD&A",
        slug="md_and_a",
    ),
    "market_risk": CanonicalSection(
        key="market_risk",
        display_name="Market Risk",
        slug="market_risk",
    ),
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

# Default sections to extract from 10-K filings
DEFAULT_SECTIONS = [
    "Business",
    "Risk Factors",
    "MD&A",
    "Market Risk",
]

def _normalize_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace("&", " and ")
    token = re.sub(r"[’'`]", "", token)
    token = re.sub(r"[^a-z0-9]+", " ", token)
    return re.sub(r"\s+", " ", token).strip()

def canonicalize_section_name(section_name: str) -> CanonicalSection | None:
    normalized = _normalize_token(section_name)

    for section_key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_token = _normalize_token(alias)
            if normalized == alias_token or normalized.startswith(f"{alias_token} "):
                return CANONICAL_SECTIONS[section_key]

    return None

def _fallback_section(section_name: str) -> CanonicalSection:
    slug = re.sub(r"[^a-z0-9]+", "_", section_name.strip().lower()).strip("_")
    key = slug or "unknown_section"
    return CanonicalSection(
        key=key,
        display_name=section_name.strip() or "Unknown Section",
        slug=key,
    )

def _clean_section_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()

def _build_document_id(ticker: str, filing_type: str, filing_date: Optional[str]) -> str:
    safe_date = filing_date or "unknown_date"
    return f"{ticker.upper()}_{filing_type}_{safe_date}"

def _build_section_lookup(filing: Filing) -> dict[str, tuple[str, Any]]:
    lookup: dict[str, tuple[str, Any]] = {}

    for raw_name, section in filing.sections.items():
        canonical = canonicalize_section_name(raw_name)
        if canonical is None:
            continue
        lookup.setdefault(canonical.key, (raw_name, section))
    return lookup

def build_index_documents(
    filing: Filing,
    sections: list[str] | None = None,
) -> BuildDocumentsResult:

    ticker = (filing.metadata.ticker or "").upper()
    filing_type = filing.metadata.filing_type
    filing_date = getattr(filing.metadata, "filing_date", None)
    company_name = getattr(filing.metadata, "company_name", None)
    accession_number = getattr(filing.metadata, "accession_number", None)
    source_url = getattr(filing.metadata, "source_url", None)

    requested_section_names = sections or DEFAULT_SECTIONS
    requested_descriptors: list[CanonicalSection] = []

    for section_name in requested_section_names:
        requested_descriptors.append(
            canonicalize_section_name(section_name) or _fallback_section(section_name)
        )

    requested_keys = [descriptor.key for descriptor in requested_descriptors]
    available_sections = _build_section_lookup(filing)
    document_id = _build_document_id(ticker, filing_type, filing_date)

    documents: list[IndexDocument] = []
    sections_found: list[str] = []
    sections_skipped: list[str] = []

    for descriptor in requested_descriptors:
        resolved = available_sections.get(descriptor.key)
        if resolved is None:
            sections_skipped.append(descriptor.key)
            continue

        raw_section_name, section_obj = resolved
        raw_text = getattr(section_obj, "content", "") or ""
        cleaned_text = _clean_section_text(raw_text)

        if not cleaned_text:
            sections_skipped.append(descriptor.key)
            continue

        chunks = [chunk.strip() for chunk in text_splitter.split_text(cleaned_text) if chunk.strip()]
        if not chunks:
            sections_skipped.append(descriptor.key)
            continue

        sections_found.append(descriptor.key)
        parent_section_id = f"{document_id}:{descriptor.slug}"

        for chunk_idx, chunk_txt in enumerate(chunks):
            chunk_id = f"{document_id}_{descriptor.slug}_{chunk_idx:03d}"
            metadata = {
                "ticker": ticker,
                "company_name": company_name,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "accession_number": accession_number,
                "source_url": source_url,
                "section": descriptor.display_name,
                "section_name": descriptor.display_name,
                "section_key": descriptor.key,
                "source_section_name": raw_section_name,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "parent_section_id": parent_section_id,
                "document_id": document_id,
            }
            documents.append(
                IndexDocument(
                    id=chunk_id,
                    text=chunk_txt,
                    metadata=metadata,
                )
            )

    return BuildDocumentsResult(
        documents=documents,
        sections_requested=requested_keys,
        sections_found=sections_found,
        sections_skipped=sections_skipped,
    )

def _count_existing_documents(store: Any, ticker: str) -> int:
    collection = getattr(store, "_collection", None)
    if collection is None:
        return 0

    existing = collection.get(where={"ticker": ticker.upper()}, include=[])
    return len(existing.get("ids", []))

def _write_documents(store: Any, documents: list[IndexDocument]) -> int:
    if not documents:
        return 0

    try:
        return store.add_documents(documents)
    except TypeError:
        return store.add_documents(
            texts=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents],
        )

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

    ticker: str = (filing.metadata.ticker or "").upper()
    filing_type = filing.metadata.filing_type

    logger.info(f"Ingesting {filing_type} for {ticker}")

    store = get_vector_store()

    if not replace_existing:
        existing_count = _count_existing_documents(store, ticker)
        if existing_count:
            logger.info(f"Chunks already present for {ticker} ({existing_count})")
            return IngestionResult(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=getattr(filing.metadata, "filing_date", None),
                total_chunks=existing_count,
                sections_processed=[],
                sections_requested=[],
                sections_found=[],
                sections_skipped=[],
                documents_written=0,
            )

    # Delete existing if requested
    if replace_existing:
        deleted = store.delete_by_ticker(ticker)
        if deleted:
            logger.info(f"Deleted {deleted} existing chunks")

    build_result = build_index_documents(filing=filing, sections=sections)

    if not build_result.documents:
        return IngestionResult(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=getattr(filing.metadata, "filing_date", None),
            error="No content found in requested sections",
            sections_requested=build_result.sections_requested,
            sections_found=build_result.sections_found,
            sections_skipped=build_result.sections_skipped,
            documents_written=0,
        )

    documents_written = _write_documents(store, build_result.documents)
    duration = datetime.now(timezone.utc) - start

    logger.info(
        "Ingested %s chunks across %s sections in %.1fs",
        documents_written,
        len(build_result.sections_found),
        duration.total_seconds(),
    )

    return IngestionResult(
        ticker=ticker,
        filing_type=filing_type,
        filing_date=getattr(filing.metadata, "filing_date", None),
        total_chunks=documents_written,
        sections_processed=build_result.sections_found,
        sections_requested=build_result.sections_requested,
        sections_found=build_result.sections_found,
        sections_skipped=build_result.sections_skipped,
        documents_written=documents_written,
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
    return ingest_filing(filing, sections=sections)

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

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\nIngesting 10-K for {ticker}...\n")

    result = await ingest_10k_for_ticker(ticker)

    if result.success:
        print(" Success!")
        print(f"   Chunks: {result.total_chunks}")
        print(f"   Sections Found: {', '.join(result.sections_found)}")
        print(f"   Sections Skipped: {', '.join(result.sections_skipped)}")
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
