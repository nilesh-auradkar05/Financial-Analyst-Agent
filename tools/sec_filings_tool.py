"""
SEC FILINGS TOOL
---------------------------------------------------------------------------

This module fetches and stores SEC filings (10-k, 10-Q) from the EDGAR system.

SEC EDGAR OVERVIEW:
---------------------------------------------------------------------------
SEC's online system for companies to submit required filings electronically.

API Requirements:
    - User-Agent header with company name and contact email
    - Max 10 requests per second

Endpoints Used:
-----------------------------
1. Company submissions: https://data.sec.gov/submissions/CIK`cik`.json
    - Returns list of all filings for a compnay

2. Filing document: https://www.sec.gov/Archives/edgar/data/`cik`/`accession`/`document`
    - Returns the actual filing document

3. Company tickers: https://www.sec.gov/files/company_tickers.json
    - Maps ticker symbols to CIK numbers

Usage:
---------------------------------------------
    from tools.sec_filings_tool import SECFilingsTool, download_10k

    # Download latest 10-K for Apple
    filing = await download_10k("AAPL")
    print(f"Downloaded: `filing.title")`
    print(f"Sections: list(filing.sections.keys())")
"""

import asyncio
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from bs4 import XMLParsedAsHTMLWarning
from langsmith import traceable
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

# Suppress XMLParsedAsHTMLWarning from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Data Models

@dataclass
class FilingMetaData:
    """
    Metadata about an SEC filing.

    This is returned when listing available filings for a company.
    Using the metadata to decide which filings to download

    cik: Central Index Key (company identifier)
        accession_number: Unique filing identifier
        filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
        filing_date: Date the filing was submitted
        report_date: Period end date the filing covers
        primary_document: Main document filename
        primary_doc_description: Description of the main document
        company_name: Name of the company
        ticker: Stock ticker symbol
    """

    cik: str
    accession_number: str
    filing_type: str
    filing_date: str
    primary_document: str

    report_date: Optional[str] = None
    primary_doc_description: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None

    @property
    def filing_url(self) -> str:
        """Construct the URL to the filing document"""
        acc_clean = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{acc_clean}/{self.primary_document}"
        )

@dataclass
class FilingSection:
    """
    A single section extracted from a filing.

    Attributes:
        name: Section name/title
        item_number: SEC item number
        content: Raw text content of the section
        word_count: Number of words in the section
    """

    name: str
    content: str

    @property
    def word_count(self) -> int:
        """Count words in the section."""
        return len(self.content.split())

@dataclass
class Filing:
    """
    Complete SEC filing with content and extracted sections.

    This is the main data structure returned after downloading
    and processing a filing.

    Attributes:
        metadata: Filing metadata (dates, accession number, etc.)
        raw_html: Original HTML content
        full_text: Extracted plain text (all content)
        sections: Dict of extracted sections by name
        tables: List of extracted tables (as text)
        local_path: Path where filing is stored locally
        downloaded_at: When the filing was downloaded
    """

    metadata: FilingMetaData
    sections: dict[str, FilingSection] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.sections) > 0

# Sec Client

class SECClient:
    """SEC EDGAR API client."""

    BASE_URL = "https://data.sec.gov"

    def __init__(self):
        self.user_agent = settings.sec.user_agent
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
            follow_redirects=True
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use 'async with SECClient():'")
        return self._client

    @traceable(name="sec_get_cik", run_type="tool", tags=["sec"])
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        try:
            # Try direct CIK lookup
            response = await self.client.get(
                f"{self.BASE_URL}/submissions/CIK{ticker.upper()}.json"
            )

            if response.status_code == 200:
                return str(response.json().get("cik", "")).zfill(10)

            # Fallback to ticker lookup
            response = await self.client.get(
                "https://www.sec.gov/files/company_tickers.json"
            )
            response.raise_for_status()

            for entry in response.json().values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)

            return None

        except Exception as e:
            logger.error(f"CIK lookup failed: {e}")
            return None

    @traceable(name="sec_get_filings", run_type="tool", tags=["sec"])
    async def get_recent_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 1,
    ) -> list[FilingMetaData]:
        """Get recent filings for a company."""
        cik = await self.get_cik(ticker)
        if not cik:
            logger.error(f"CIK not found for {ticker}")
            return []

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/submissions/CIK{cik}.json"
            )
            response.raise_for_status()
            data = response.json()

            company_name = data.get("name", ticker)
            filings = data.get("filings", {}).get("recent", {})

            results = []
            forms = filings.get("form", [])

            for i, form in enumerate(forms):
                if form == filing_type and len(results) < count:
                    results.append(FilingMetaData(
                        cik=cik,
                        accession_number=filings["accessionNumber"][i],
                        filing_type=form,
                        filing_date=filings["filingDate"][i],
                        report_date=filings.get("reportDate", [None] * len(forms))[i],
                        primary_document=filings["primaryDocument"][i],
                        company_name=company_name,
                        ticker=ticker.upper(),
                    ))

            return results

        except Exception as e:
            logger.error(f"Filing lookup failed: {e}")
            return []

    @traceable(name="sec_download_filing", run_type="tool", tags=["sec"])
    async def download_filing(self, metadata: FilingMetaData) -> Filing:
        """Download and parse a filing."""
        logger.info(f"Downloading {metadata.filing_type} for {metadata.ticker}")

        try:
            response = await self.client.get(metadata.filing_url)
            response.raise_for_status()
            raw_text = response.text

            # Parse sections
            sections = self._parse_sections(raw_text)

            return Filing(metadata=metadata, sections=sections)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return Filing(metadata=metadata, error=str(e))

    def _parse_sections(self, text: str) -> dict[str, FilingSection]:
        """Extract sections from filing HTML/text."""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean)

        sections = {}

        # Section patterns for 10-K
        patterns = {
            "Business": r"Item\s*1[.\s]+Business",
            "Risk Factors": r"Item\s*1A[.\s]+Risk\s*Factors",
            "MD&A": r"Item\s*7[.\s]+Management",
            "Market Risk": r"Item\s*7A[.\s]+Quantitative",
        }

        for name, pattern in patterns.items():
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                start = match.start()

                # Find next section
                end = len(clean)
                for other_pattern in patterns.values():
                    if other_pattern != pattern:
                        other = re.search(other_pattern, clean[start + 100:], re.IGNORECASE)
                        if other:
                            end = min(end, start + 100 + other.start())

                content = clean[start:end].strip()

                if len(content) > 500:  # Meaningful content
                    sections[name] = FilingSection(
                        name=name,
                        content=content[:50000],  # Limit size
                    )

        return sections


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@traceable(name="get_latest_10k", run_type="tool", tags=["sec", "10k"])
async def get_latest_10k(
    ticker: str,
) -> Optional[Filing]:
    """
    Download the latest 10-K for a company (convenience function).

    Args:
        ticker: Stock ticker symbol
        save_locally: Whether to save to disk

    Returns:
        Filing object or None if not found

    Example:
        filing = await download_10k("AAPL")
        if filing:
            print(filing.get_section("Risk Factors").content[:500])
    """
    async with SECClient() as client:
        filings = await client.get_recent_filings(ticker, filing_type="10-K", count=1)

        if not filings:
            logger.warning(f"No 10-K filings found for {ticker}")
            return None

        return await client.download_filing(filings[0])

# =============================================================================
# CLI / TESTING
# =============================================================================


async def _main():
    """Test the SEC filings tool."""
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\nFetching 10-K for {ticker}...\n")

    filing = await get_latest_10k(ticker)

    if filing and filing.success:
        print(f"Company: {filing.metadata.company_name}")
        print(f"Filing Date: {filing.metadata.filing_date}")
        print(f"Sections: {list(filing.sections.keys())}")

        for name, section in filing.sections.items():
            print(f"\n{name}: {section.word_count} words")
            print(f"Preview: {section.content[:200]}...")
    else:
        error = filing.error if filing else "Filing not found"
        print(f"Error: {error}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    # Run async main
    asyncio.run(_main())
