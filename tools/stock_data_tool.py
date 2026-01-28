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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langsmith import traceable
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

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
        stock = yf.Ticker(ticker)
        info = stock.info

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


# =============================================================================
# CLI
# =============================================================================


async def _main():
    """Test stock data fetching."""
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\nFetching data for {ticker}...\n")

    info = await get_stock_data(ticker)

    if info.success:
        print(info.to_summary())
        print(f"\n52-Week Range: ${info.fifty_two_week_low} - ${info.fifty_two_week_high}")
        print(f"Industry: {info.industry}")
        print(f"Recommendation: {info.recommendation}")
    else:
        print(f"Error: {info.error}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
