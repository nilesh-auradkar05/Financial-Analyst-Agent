"""
STOCK DATA TOOL

This module provides stock market data using the Yfinance library.

Data Fetched:
1. Current Quote: price, open, close, volume, day's range.
2. Company info: market cap, P/E ratio, sector, description.
3. Price History: historical data for trend analysis.
4. Analyst Targets: 1-yr price target estimates

Use:
-------------------
    from tools.stock_data_tool import get_stock_quote, get_company_info

    # Get current quote
    quote = get_stock_quote("AAPL")
    print(f"Apple: $`quote.current_price:.2f`")

    # get company info
    info = get_company_info("AAPL")
    print(f"Market Cap: $`info.market_cap:,`")
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from langsmith import traceable
from loguru import logger

from configs.config import settings

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class StockInfo:
    """Stock market data."""
    ticker: str
    company_name: str
    current_price: Optional[float] = None
    price_change_percent: Optional[float] = None
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    dividend_yield: Optional[float] = None
    target_price: Optional[float] = None
    recommendation: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.current_price is not None

    def to_summary(self) -> str:
        """Format as text summary."""
        parts = [f"{self.company_name} ({self.ticker})"]

        if self.current_price:
            parts.append(f"Price: ${self.current_price:.2f}")
        if self.market_cap:
            if self.market_cap >= 1e12:
                parts.append(f"Market Cap: ${self.market_cap/1e12:.2f}T")
            else:
                parts.append(f"Market Cap: ${self.market_cap/1e9:.2f}B")
        if self.pe_ratio:
            parts.append(f"P/E: {self.pe_ratio:.1f}")
        if self.sector:
            parts.append(f"Sector: {self.sector}")

        return " | ".join(parts)

def _yfinance_fetch_sync(ticker: str) -> dict:
    """
    Sync fetch - runs in a thread via asyncio.to_thread

    yfinance is fully synchronous and makes HTTP calls internally.
    We wrap it in a thread to avoid blocking the event loop.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed")
    stock = yf.Ticker(ticker)
    return dict(stock.info)

def _with_retry(func):
    """Apply tenacity retry if available."""
    if not TENACITY_AVAILABLE:
        return func

    return retry(
        stop=stop_after_attempt(settings.retry.max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry.min_wait_seconds,
            max=settings.retry.max_wait_seconds,
        ),
        reraise=True,
    )(func)

# =============================================================================
# STOCK DATA FUNCTION
# =============================================================================


@traceable(name="get_stock_data", run_type="tool", tags=["stock", "yfinance"])
async def get_stock_data(ticker: str) -> StockInfo:
    """
    Fetch stock data from YFinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        StockInfo with current market data
    """
    if not YFINANCE_AVAILABLE or yf is None:
        return StockInfo(
            ticker=ticker,
            company_name=ticker,
            error="yfinance not installed",
        )

    logger.info(f"Fetching stock data for {ticker}")

    try:
        fetcher = _with_retry(_yfinance_fetch_sync)
        info = await asyncio.to_thread(fetcher, ticker)

        stock_info = StockInfo(
            ticker=ticker.upper(),
            company_name=info.get("longName") or info.get("shortName") or ticker,
            current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
            price_change_percent=info.get("regularMarketChangePercent"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            volume=info.get("volume"),
            avg_volume=info.get("averageVolume"),
            dividend_yield=info.get("dividendYield"),
            target_price=info.get("targetMeanPrice"),
            recommendation=info.get("recommendationKey"),
            sector=info.get("sector"),
            industry=info.get("industry"),
        )
        logger.info(f"{ticker}: ${stock_info.current_price}")
        return stock_info

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return StockInfo(ticker=ticker, company_name=ticker, error=str(e))
