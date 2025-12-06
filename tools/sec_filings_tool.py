f"""
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
from optparse import Option
import httpx
import json
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

# Suppress XMLParsedAsHTMLWarning from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings, FILINGS_DIR

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
    def accession_number_raw(self) -> str:
        """Accession number"""
        return self.accession_number.replace("-", "")

    @property
    def filing_url(self) -> str:
        """Construct the URL to the filing document"""
        cik_padded = self.cik.zfill(10)
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_number_raw}/{self.primary_document}"
        )

    @property
    def index_url(self) -> str:
        """URL to the filing index page."""
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_number_raw}/"
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
    item_number: Optional[str] = None

    @property
    def word_count(self) -> int:
        """Count words in the section."""
        return len(self.content.split())

    def to_context(self) -> str:
        """Format section for LLM context."""
        header = f"=== {self.name} ==="
        if self.item_number:
            header = f"=== Item {self.item_number}: {self.name} ==="
        return f"{header}\n\n{self.content}"

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
    raw_html: str
    full_text: str
    sections: dict[str, FilingSection] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)
    local_path: Optional[Path] = None
    downloaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def ticker(self) -> str:
        """Get ticker from metadata"""
        return self.metadata.ticker or "UNKNOWN"

    @property
    def filing_type(self) -> str:
        """Get filing type from metadata"""
        return self.metadata.filing_type

    @property
    def filing_date(self) -> str:
        """Get filing date from metadata"""
        return self.metadata.filing_date

    def get_section(self, name: str) -> Optional[FilingSection]:
        """
        Get a section by name.

        Args:
            name: Section name to find ("Risk Factors", "risk factors)

        Returns:
            FilingSection if found, else None
        """
        name_lower = name.lower()
        for section_name, section in self.sections.items():
            if name_lower in section_name.lower():
                return section
        return None

    def to_context(self, include_sections: Optional[list[str]] = None) -> str:
        """
        Format filing as context for LLM.
        
        Args:
            include_sections: List of section names to include.
                            If None, includes all sections.
        
        Returns:
            Formatted string for LLM context
        """
        lines = [
            f"=== SEC Filing: {self.metadata.filing_type} ===",
            f"Company: {self.metadata.company_name} ({self.ticker})",
            f"Filing Date: {self.filing_date}",
            f"Report Dated: {self.metadata.report_date or 'N/A'}",
            "",
        ]

        for name, section in self.sections.items():
            if include_sections is None or name in include_sections:
                lines.append(section.to_context())
                lines.append("")

        return "\n".join(lines)

@dataclass
class SECSearchResult:
    """
    Result of searching for filings.
    
    Attributes:
        ticker: Ticker that was searched
        company_name: Company name
        cik: Central Index Key
        filings: List of available filings
        error: Error message if search failed
    """
    
    ticker: str
    company_name: Optional[str] = None
    cik: Optional[str] = None
    filings: list[FilingMetaData] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Did the search succeed?"""
        return self.error is None and self.cik is not None
    
    @property
    def has_filings(self) -> bool:
        """Were any filings found?"""
        return len(self.filings) > 0

# SEC Filings Tool

class SECFilingsTool:
    """
    Tool for fetching SEC filings from EDGAR

    This class handles:
        - Ticker to CIK mapping
        - Filing discovery and metadata retrieval
        - Document downloading
        - HTML parsing and text extraction
        - Section extraction (Risk Factors, etc.)
        - Local storage management

    Example:
    -----------------------------------
        async with SECFilingsTool() as tool:
            # Search for filings
            result = await tool.search_filings("AAPL", filing_types="10-K")

            # Download the latest 10-K
            if result.has_filings:
                filing = await tool.download_filing(result.filings[0])
                print(filing.sections.keys())
    """

    SECTIONS_10K = {
        "1": "Business",
        "1A": "Risk Factors",
        "1B": "Unresolved Staff Comments",
        "2": "Properties",
        "3": "Legal Proceedings",
        "4": "Mine Safety Disclosures",
        "5": "Market for Registrant's Common Equity",
        "6": "Selected Financial Data",
        "7": "Management's Discussion and Analysis",
        "7A": "Quantitative and Qualitative Disclosures About Market Risk",
        "8": "Financial Statements and Supplementary Data",
        "9": "Changes in and Disagreements with Accountants",
        "9A": "Controls and Procedures",
        "9B": "Other Information",
        "10": "Directors, Executive Officers and Corporate Governance",
        "11": "Executive Compensation",
        "12": "Security Ownership",
        "13": "Certain Relationships and Related Transactions",
        "14": "Principal Accountant Fees and Services",
    }

    # Sections most useful for investment analysis
    KEY_SECTIONS = ["1", "1A", "7", "7A", "8"]

    def __init__(
        self,
        user_agent: Optional[str] = None,
        storage_dir: Optional[Path] = None,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the SEC filings tool.
        
        Args:
            user_agent: User-Agent header (SEC requirement).
                       If not provided, uses settings.sec.user_agent
            storage_dir: Directory to store downloaded filings.
                        If not provided, uses FILINGS_DIR from config.
            rate_limit_delay: Delay between requests (limit of max 10/sec)
        """
        self.user_agent = user_agent or settings.sec.user_agent
        self.storage_dir = storage_dir or FILINGS_DIR
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client (created in __aenter__)
        self._client: Optional[httpx.AsyncClient] = None
        
        # Cache for ticker -> CIK mapping
        self._ticker_to_cik: dict[str, str] = {}
        
        logger.info(
            f"SECFilingsTool initialized "
            f"(storage: {self.storage_dir}, rate_limit: {self.rate_limit_delay}s)"
        )
    
    async def __aenter__(self):
        """Async context manager entry - create HTTP client."""
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
            follow_redirects=True,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError(
                "HTTP client not initialized. Use 'async with SECFilingsTool() as tool:'"
            )
        return self._client
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        await asyncio.sleep(self.rate_limit_delay)
    
    async def _fetch(self, url: str) -> httpx.Response:
        """pyth
        Fetch a URL with rate limiting and error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            httpx.Response object
            
        Raises:
            httpx.HTTPError: On network errors
        """
        await self._rate_limit()
        logger.debug(f"Fetching: {url}")
        
        response = await self.client.get(url)
        response.raise_for_status()
        
        return response
    
    # =========================================================================
    # TICKER TO CIK MAPPING
    # =========================================================================
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker symbol.
        
        SEC uses CIK as the primary identifier, but most users
        know tickers. This method bridges that gap.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            CIK string (e.g., "320193") or None if not found
        """
        ticker = ticker.upper().strip()
        
        # Check cache first
        if ticker in self._ticker_to_cik:
            return self._ticker_to_cik[ticker]
        
        logger.info(f"Looking up CIK for {ticker}")
        
        try:
            # SEC provides a JSON file mapping tickers to CIKs
            url = "https://www.sec.gov/files/company_tickers.json"
            response = await self._fetch(url)
            data = response.json()
            
            # Data format: {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc"}, ...}
            for entry in data.values():
                entry_ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", ""))
                
                # Cache all entries while we're at it
                self._ticker_to_cik[entry_ticker] = cik
                
                if entry_ticker == ticker:
                    logger.info(f"Found CIK for {ticker}: {cik}")
                    return cik
            
            logger.warning(f"CIK not found for ticker: {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to lookup CIK for {ticker}: {e}")
            return None
    
    # =========================================================================
    # FILING SEARCH
    # =========================================================================
    
    async def search_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 5,
    ) -> SECSearchResult:
        """
        Search for filings by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing to search for ("10-K", "10-Q", etc.)
            count: Maximum number of filings to return
            
        Returns:
            SECSearchResult with list of FilingMetadata
        """
        ticker = ticker.upper().strip()
        logger.info(f"Searching {filing_type} filings for {ticker}")
        
        # Get CIK
        cik = await self.get_cik(ticker)
        if not cik:
            return SECSearchResult(
                ticker=ticker,
                error=f"Could not find CIK for ticker: {ticker}",
            )
        
        try:
            # Fetch company submissions
            # CIK must be zero-padded to 10 digits
            cik_padded = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            
            response = await self._fetch(url)
            data = response.json()
            
            company_name = data.get("name", "")
            
            # Parse recent filings
            recent = data.get("filings", {}).get("recent", {})
            
            # These are parallel arrays - same index across all
            forms = recent.get("form", [])
            accession_numbers = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            primary_documents = recent.get("primaryDocument", [])
            primary_doc_descriptions = recent.get("primaryDocDescription", [])
            
            # Filter by filing type and build metadata objects
            filings = []
            for i in range(len(forms)):
                if forms[i] == filing_type:
                    filings.append(
                        FilingMetaData(
                            cik=cik,
                            accession_number=accession_numbers[i],
                            filing_type=forms[i],
                            filing_date=filing_dates[i],
                            report_date=report_dates[i] if i < len(report_dates) else None,
                            primary_document=primary_documents[i],
                            primary_doc_description=(
                                primary_doc_descriptions[i] 
                                if i < len(primary_doc_descriptions) 
                                else None
                            ),
                            company_name=company_name,
                            ticker=ticker,
                        )
                    )
                    
                    if len(filings) >= count:
                        break
            
            logger.info(f"Found {len(filings)} {filing_type} filings for {ticker}")
            
            return SECSearchResult(
                ticker=ticker,
                company_name=company_name,
                cik=cik,
                filings=filings,
            )
            
        except Exception as e:
            logger.error(f"Failed to search filings for {ticker}: {e}")
            return SECSearchResult(
                ticker=ticker,
                cik=cik,
                error=str(e),
            )
    
    # =========================================================================
    # FILING DOWNLOAD & PARSING
    # =========================================================================
    
    async def download_filing(
        self,
        metadata: FilingMetaData,
        extract_sections: bool = True,
        save_locally: bool = True,
    ) -> Filing:
        """
        Download and parse a filing.
        
        Args:
            metadata: FilingMetadata from search_filings()
            extract_sections: Whether to extract individual sections
            save_locally: Whether to save the filing to disk
            
        Returns:
            Filing object with content and extracted sections
        """
        logger.info(
            f"Downloading {metadata.filing_type} for {metadata.ticker} "
            f"(filed: {metadata.filing_date})"
        )
        
        try:
            # Fetch the filing document
            response = await self._fetch(metadata.filing_url)
            raw_html = response.text
            
            # Parse HTML and extract text
            full_text = self._extract_text(raw_html)
            
            # Extract sections if requested
            sections = {}
            if extract_sections and metadata.filing_type == "10-K":
                sections = self._extract_sections(raw_html, full_text)
            
            # Extract tables
            tables = self._extract_tables(raw_html)
            
            # Save locally if requested
            local_path = None
            if save_locally:
                local_path = self._save_filing(metadata, raw_html, full_text)
            
            filing = Filing(
                metadata=metadata,
                raw_html=raw_html,
                full_text=full_text,
                sections=sections,
                tables=tables,
                local_path=local_path,
            )
            
            logger.info(
                f"Downloaded {metadata.filing_type}: "
                f"{len(full_text)} chars, {len(sections)} sections extracted"
            )
            
            return filing
            
        except Exception as e:
            logger.error(f"Failed to download filing: {e}")
            raise
    
    def _extract_text(self, html: str) -> str:
        """
        Extract plain text from HTML filing.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Cleaned plain text
        """
        soup = BeautifulSoup(html, "lxml")
        
        # Remove script and style elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        
        return "\n".join(lines)
    
    def _extract_sections(
        self,
        html: str,
        full_text: str,
    ) -> dict[str, FilingSection]:
        """
        Extract standard 10-K sections from the filing.
        
        This uses regex patterns to find section headers and extract
        content between them. SEC filings follow a standard format,
        but there's variation in how companies format headers.
        
        Args:
            html: Raw HTML content
            full_text: Extracted plain text
            
        Returns:
            Dict mapping section names to FilingSection objects
        """
        sections = {}
        
        # Pattern to match "Item 1A" style headers
        # Handles variations like "ITEM 1A", "Item 1A.", "Item 1A -", etc.
        item_pattern = re.compile(
            r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-B]?)\.?\s*[-–—:]?\s*([^\n]+)?",
            re.MULTILINE
        )
        
        # Find all item headers
        matches = list(item_pattern.finditer(full_text))
        
        for i, match in enumerate(matches):
            item_number = match.group(1).upper()
            item_title = match.group(2) or ""
            item_title = item_title.strip().rstrip(".")
            
            # Get standard section name if we recognize the item number
            if item_number in self.SECTIONS_10K:
                section_name = self.SECTIONS_10K[item_number]
            else:
                section_name = item_title or f"Item {item_number}"
            
            # Extract content between this header and the next
            start_pos = match.end()
            
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(full_text)
            
            content = full_text[start_pos:end_pos].strip()
            
            # Only include sections with meaningful content
            if len(content) > 100:
                sections[section_name] = FilingSection(
                    name=section_name,
                    content=content,
                    item_number=item_number,
                )
        
        logger.debug(f"Extracted {len(sections)} sections: {list(sections.keys())}")
        
        return sections
    
    def _extract_tables(self, html: str, max_tables: int = 20) -> list[str]:
        """
        Extract tables from HTML as text.
        
        Financial statements are often in tables. This extracts them
        as plain text for potential analysis.
        
        Args:
            html: Raw HTML content
            max_tables: Maximum number of tables to extract
            
        Returns:
            List of tables as formatted text strings
        """
        soup = BeautifulSoup(html, "lxml")
        tables = []
        
        for table in soup.find_all("table")[:max_tables]:
            rows = []
            for tr in table.find_all("tr"):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    cell_text = td.get_text(strip=True)
                    cells.append(cell_text)
                if cells:
                    rows.append(" | ".join(cells))
            
            if rows:
                tables.append("\n".join(rows))
        
        return tables
    
    def _save_filing(
        self,
        metadata: FilingMetaData,
        raw_html: str,
        full_text: str,
    ) -> Path:
        """
        Save filing to local storage.
        
        Creates files:
        - {ticker}_{type}_{date}.html (raw HTML)
        - {ticker}_{type}_{date}.txt (extracted text)
        
        Args:
            metadata: Filing metadata
            raw_html: Raw HTML content
            full_text: Extracted plain text
            
        Returns:
            Path to the saved text file
        """
        # Create filename
        base_name = f"{metadata.ticker}_{metadata.filing_type}_{metadata.filing_date}"
        base_name = base_name.replace("/", "-")  # Sanitize
        
        html_path = self.storage_dir / f"{base_name}.html"
        text_path = self.storage_dir / f"{base_name}.txt"
        
        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(raw_html)
        
        # Save text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        logger.info(f"Saved filing to: {text_path}")
        
        return text_path
    
    # =========================================================================
    # HIGH-LEVEL CONVENIENCE METHODS
    # =========================================================================
    
    async def get_latest_10k(
        self,
        ticker: str,
        extract_sections: bool = True,
        save_locally: bool = True,
    ) -> Optional[Filing]:
        """
        Get the most recent 10-K filing for a company.
        
        This is the main method for fetching annual reports.
        
        Args:
            ticker: Stock ticker symbol
            extract_sections: Whether to extract individual sections
            save_locally: Whether to save to disk
            
        Returns:
            Filing object or None if not found
        """
        # Search for 10-K filings
        result = await self.search_filings(ticker, filing_type="10-K", count=1)
        
        if not result.success:
            logger.error(f"Failed to search filings: {result.error}")
            return None
        
        if not result.has_filings:
            logger.warning(f"No 10-K filings found for {ticker}")
            return None
        
        # Download the latest one
        return await self.download_filing(
            result.filings[0],
            extract_sections=extract_sections,
            save_locally=save_locally,
        )
    
    async def get_key_sections(
        self,
        ticker: str,
    ) -> dict[str, FilingSection]:
        """
        Get just the key sections from the latest 10-K.
        
        Key sections for investment analysis:
        - Business (Item 1)
        - Risk Factors (Item 1A)
        - MD&A (Item 7)
        - Market Risk (Item 7A)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict of section name -> FilingSection
        """
        filing = await self.get_latest_10k(ticker)
        
        if not filing:
            return {}
        
        # Filter to key sections
        key_sections = {}
        for item_num in self.KEY_SECTIONS:
            section_name = self.SECTIONS_10K.get(item_num, "")
            if section_name in filing.sections:
                key_sections[section_name] = filing.sections[section_name]
        
        return key_sections


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def download_10k(
    ticker: str,
    save_locally: bool = True,
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
    async with SECFilingsTool() as tool:
        return await tool.get_latest_10k(ticker, save_locally=save_locally)


async def get_company_risk_factors(ticker: str) -> Optional[str]:
    """
    Get just the Risk Factors section from latest 10-K.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Risk factors text or None if not found
    """
    async with SECFilingsTool() as tool:
        filing = await tool.get_latest_10k(ticker)
        
        if filing:
            section = filing.get_section("Risk Factors")
            if section:
                return section.content
        
        return None


async def search_sec_filings(
    ticker: str,
    filing_type: str = "10-K",
    count: int = 5,
) -> SECSearchResult:
    """
    Search for SEC filings (convenience function).
    
    Args:
        ticker: Stock ticker symbol
        filing_type: Type of filing ("10-K", "10-Q", "8-K", etc.)
        count: Maximum number of results
        
    Returns:
        SECSearchResult with list of FilingMetadata
    """
    async with SECFilingsTool() as tool:
        return await tool.search_filings(ticker, filing_type, count)


# =============================================================================
# CLI / TESTING
# =============================================================================


async def _main():
    """Test the SEC filings tool."""
    import sys
    
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"\n{'='*60}")
    print(f"SEC Filings Tool - Testing with {ticker}")
    print(f"{'='*60}\n")
    
    async with SECFilingsTool() as tool:
        # Search for filings
        print("Searching for 10-K filings...")
        result = await tool.search_filings(ticker, filing_type="10-K", count=3)
        
        if not result.success:
            print(f"Search failed: {result.error}")
            return
        
        print(f"Found {len(result.filings)} filings for {result.company_name}")
        print(f"   CIK: {result.cik}\n")
        
        for i, filing in enumerate(result.filings, 1):
            print(f"   {i}. {filing.filing_type} - Filed: {filing.filing_date}")
            print(f"      Report Period: {filing.report_date}")
            print(f"      Document: {filing.primary_document}")
            print()
        
        # Download the latest one
        if result.has_filings:
            print("Downloading latest 10-K...")
            filing = await tool.download_filing(result.filings[0])
            
            print(f"\nDownloaded: {filing.metadata.filing_type}")
            print(f"   Total text length: {len(filing.full_text):,} characters")
            print(f"   Tables extracted: {len(filing.tables)}")
            print(f"   Sections extracted: {len(filing.sections)}")
            print(f"   Saved to: {filing.local_path}")
            
            print("\n   Sections found:")
            for name, section in filing.sections.items():
                print(f"   - {name}: {section.word_count:,} words")
            
            # Show snippet of Risk Factors
            risk_factors = filing.get_section("Risk Factors")
            if risk_factors:
                print(f"\n{'─'*60}")
                print("Risk Factors Preview (first 500 chars):")
                print(f"{'─'*60}")
                print(risk_factors.content[:500] + "...")


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
