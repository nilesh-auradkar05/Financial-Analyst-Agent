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
    from app.services.tools.sec_filings_tool import SECFilingsTool, download_10k

    # Download latest 10-K for Apple
    filing = await download_10k("AAPL")
    print(f"Downloaded: `filing.title")`
    print(f"Sections: list(filing.sections.keys())")
"""

import asyncio
import html
import re
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Optional, cast

import httpx
from bs4 import XMLParsedAsHTMLWarning
from langsmith import traceable
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

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

def _filing_from_edgartools_payload(payload: Any) -> Filing:
    """Adapt EdgarTools payload into this module's existing Filing contract."""
    metadata = FilingMetaData(
        cik=payload.cik,
        accession_number=payload.accession_number,
        filing_type=payload.filing_type,
        filing_date=payload.filing_date,
        report_date=payload.report_date,
        primary_document=payload.primary_document,
        company_name=payload.company_name,
        ticker=payload.ticker,
    )

    sections = {
        name: FilingSection(name=name, content=section.content)
        for name, section in payload.sections.items()
    }

    return Filing(metadata=metadata, sections=sections)


async def _get_latest_10k_with_edgartools(ticker: str) -> Optional[Filing]:
    """Try EdgarTools extraction without blocking the event loop."""
    try:
        extractor_module = import_module("app.services.tools.edgartools_sec_extractor")
        extract_latest_10k_with_edgartools = getattr(
            extractor_module,
            "extract_latest_10k_with_edgartools",
        )
        payload = await asyncio.to_thread(
            extract_latest_10k_with_edgartools,
            ticker,
            identity=settings.sec.user_agent,
        )
    except Exception as exc:
        logger.warning("EdgarTools 10-K extraction failed for %s: %s", ticker, exc)
        return None

    filing = _filing_from_edgartools_payload(payload)

    logger.info(
        "EdgarTools extracted %s sections for %s: %s",
        len(filing.sections),
        ticker,
        list(filing.sections.keys()),
    )

    return filing

def _http_retry(func):
    """Apply tenacity retry decorator to httpx calls if available."""
    return retry(
        stop=stop_after_attempt(settings.retry.max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry.min_wait_seconds,
            max=settings.retry.max_wait_seconds,
        ),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.HTTPStatusError),
        ),
        reraise=True,
    )(func)



# Sec Client

class SECClient:
    """SEC EDGAR API client."""

    BASE_URL = "https://data.sec.gov"

    def __init__(self):
        self.user_agent = settings.sec.user_agent
        self._timeout = settings.retry.http_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=self._timeout,
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

    @_http_retry
    async def _get(self, url: str) -> httpx.Response:
        """Single HTTP GET with retry."""
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp

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
            response = await self._get(
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
            response = await self._get(
                f"{self.BASE_URL}/submissions/CIK{cik}.json"
            )
            response.raise_for_status()
            data = response.json()

            company_name = data.get("name", ticker)
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])

            results: list[FilingMetaData] = []
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
            response = await self._get(metadata.filing_url)
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
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = html.unescape(clean)
        clean = re.sub(r'\s+', ' ', clean)

        patterns = {
            "Business": r"Item\s*1[.\s]+Business",
            "Risk Factors": r"Item\s*1A[.\s]+Risk\s*Factors",
            "MD&A": r"Item\s*7[.\s]+Management",
            "Market Risk": r"Item\s*7A[.\s]+Quantitative",
        }

        all_starts: list[tuple[str, int]] = []
        for name, pattern in patterns.items():
            for m in re.finditer(pattern, clean, re.IGNORECASE):
                all_starts.append((name, m.start()))
        all_starts.sort(key=lambda t: t[1])

        sections: dict[str, FilingSection] = {}
        for idx, (name, start) in enumerate(all_starts):
            end = all_starts[idx + 1][1] if idx + 1 < len(all_starts) else len(clean)
            content = clean[start:end].strip()

            if len(content) <= 500:
                continue

            if name not in sections or len(content) > len(sections[name].content):
                sections[name] = FilingSection(
                    name=name,
                    content=content[:50000],
                )

        return sections


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@traceable(name="get_latest_10k", run_type="tool", tags=["sec", "10k"])
async def get_latest_10k(
    ticker: str,
) -> Optional[Filing]:
    """Download the latest 10-K for a company.

    Primary path:
        EdgarTools section extraction.

    Fallback path:
        Existing SECClient download + local regex parser.

    This keeps the public Filing contract stable for ingestion while letting us
    test whether EdgarTools fixes missing section extraction for MSFT/NVDA.
    """
    normalized_ticker = ticker.upper().strip()

    edgar_filing = await _get_latest_10k_with_edgartools(normalized_ticker)
    if edgar_filing is not None and edgar_filing.sections:
        return edgar_filing

    logger.warning(
        "Falling back to existing SECClient parser for %s because EdgarTools did not return sections",
        normalized_ticker,
    )

    async with SECClient() as client:
        filings = await client.get_recent_filings(
            normalized_ticker,
            filing_type="10-K",
            count=1,
        )

        if not filings:
            logger.warning("No 10-K filings found for %s", normalized_ticker)
            return None

        return cast(Optional[Filing], await client.download_filing(filings[0]))
