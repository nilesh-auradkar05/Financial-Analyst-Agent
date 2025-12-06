f"""
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

from math import inf
from multiprocessing import Value
from optparse import Option
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import settings

# Data Models

@dataclass
class StockQuote:
    """
    Current Stock quote with price and trading data.

    Attributes:
        ticker: Stock symbol (eg: AAPL)
        current_price: Latest trading price
        open_price: Price at market open
        previous_close: Yesterday's closing price
        day_high: Highest price today
        day_low: Lowest price today
        volume: Number of shares traded today
        average_volume: Average daily volume
        fifty_two_week_high: Highest price in past 52 weeks
        fifty_two_week_low: Lowest price in past 52 weeks
        timestamp: When this quote was fetched
    """

    ticker: str = None

    # Price data identifier
    current_price: float = None
    open_price: Optional[float] = None
    previous_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None

    # Volume data identifiers
    volume: Optional[int] = None
    average_volume: Optional[int] = None
    
    # 52 Week prices
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

    # METADATA
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def day_range(self) -> str:
        """Format day's trading range as string."""
        if self.day_low and self.day_high:
            return f"${self.day_low:.2f} - ${self.day_high:.2f}"
        return "N/A"

    @property
    def fifty_two_week_range(self) -> str:
        """Format 52-week range as String."""
        if self.fifty_two_week_low and self.fifty_two_week_high:
            return f"${self.fifty_two_week_low:.2f} - ${self.fifty_two_week_high:.2f}"
        return "N/A"

    @property
    def day_change(self) -> Optional[float]:
        """Calculate dollar change from previous close."""
        if self.previous_close and self.current_price:
            return self.current_price - self.previous_close
        return None

    @property
    def day_change_percent(self) -> Optional[float]:
        """Calculate percentage change from previous close."""

        if self.previous_close and self.current_price:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100

        return None
    
    def to_summary(self) -> str:
        """
        Format quote as a human-readable summary for the memo.

        Returns:
            Multi-line string with key quote data
        """
        change = self.day_change
        change_pct = self.day_change_percent
        change_str = ""

        if change is not None and change_pct is not None:
            sign = "+" if change >= 0 else ""
            change_str = f" ({sign}{change:.2f}, {sign}{change_pct:.2f}%)"

        lines = [
            f"Current Price: ${self.current_price:.2f}{change_str}",
            f"Day's Range: {self.day_range}",
            f"52-Week Range: {self.fifty_two_week_range}",
            f"Volume: {self.volume:,}" if self.volume else "Volume: N/A",
        ]

        return "\n".join(lines)

@dataclass
class CompanyInfo:
    f"""
    Company fundamental information and profile.

    This provides context about what the company is and its
    key financial metrics. Used in the MEMO's company overview section.

    Attributes:
        ticker: Stock Symbol
        name: Full Company Name
        sector: Business Sector (eg: Technology)
        industry: Specific Industry (eg: Consumer Electronics)
        description: Business description/summary
        market_cap: Total market capitalization
        pe_ratio: Price-to-Earnings ratio
        forward_pe: Forward P/E ratio
        eps: Earnings per share
        dividend_yield: Annual dividend yeild as percentage
        target_price: Analyst 1-yr target price
        target_high: Highest Analyst target
        target_low: Lowest analyst target
        recommendation: Analyst Recommendation (eg: "buy", "hold", "sell")
        website: Company website URL
        employees: Number of full-time Employees
    """

    ticker: str
    name: str

    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None

    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    eps: Optional[float] = None

    dividend_yield: Optional[float] = None

    target_price: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    recommendation: Optional[str] = None

    website: Optional[str] = None
    employees: Optional[int] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def market_cap_formatted(self) -> str:
        """Format market cap in human-readable form"""
        if not self.market_cap:
            return "N/A"

        if self.market_cap >= 1_000_000_000_000:
            return f"${self.market_cap / 1_000_000_000_000:.2f}T"
        elif self.market_cap >= 1_000_000_000:
            return f"${self.market_cap / 1_000_000_000:.2f}B"
        elif self.market_cap >= 1_000_000:
            return f"${self.market_cap / 1_000_000:.2f}M"
        else:
            return f"${self.market_cap:,}"

    def to_summary(self) -> str:
        """
        Format company info as a summary for the memo.

        Returns:
            Multi-line string with key company data
        """
        lines = [
            f"Company: {self.name} ({self.ticker})",
            f"Sector: {self.sector or 'N/A'} | Industry: {self.industry or 'N/A'}",
            f"Market Cap: {self.market_cap_formatted}",
            f"P/E Ratio: {self.pe_ratio:.2f}" if self.pe_ratio else "P/E Ratio: N/A",
            f"EPS (TTM): ${self.eps:.2f}" if self.eps else "EPS: N/A",
        ]

        if self.target_price:
            lines.append(
                f"1Y Target: ${self.target_price:.2f}"
                f"(Range: ${self.target_low:.2f} - ${self.target_high:.2f})"
                if self.target_low and self.target_high
                else f"1Y Target: ${self.target_price:.2f}"
            )

        if self.recommendation:
            lines.append(f"Analyst Rating: {self.recommendation.upper()}")

        return "\n".join(lines)

    def to_context(self) -> str:
        """
        Format as context for LLM analysis.

        Returns:
            Structured string for LM context window
        """
        return f"""
        Company: {self.name} ({self.ticker})
        Sector: {self.sector or 'Unknown'}
        Industry: {self.industry or 'Unknown'}
        
        Business Description:
            {self.description or 'No description available.'}

        Key Metrics:
            - Market Cap: {self.market_cap_formatted}
            - P/E Ratio: {f"{self.pe_ratio:.2f}" if self.pe_ratio is not None else 'N/A'}
            - Forward P/E: {f"{self.forward_pe:.2f}" if self.forward_pe is not None else 'N/A'}
            - EPS (TTM): {f"{self.eps:.2f}" if self.eps is not None else 'N/A'}
            - Dividend Yield: {f"{self.dividend_yield:.2f}" if self.dividend_yield is not None else 'N/A'} 
            - Target Price: ${f"{self.target_price:.2f}" if self.target_price is not None else 'N/A'}
            - Recommendation: {self.recommendation or 'N/A'}
        """

@dataclass
class PriceHistory:
    """
    Historical price data for trend Analysis.

    Contains Open, high, low, close, volume data points
    for a specified period of time

    Atrributes:
        ticker: Stock Symbol
        period: Time period requested (eg: 1Y, 1Mo)
        data_points: List of daily OHLCV records
        start_date: First date in the history
        end_date: Last date in history
    """

    ticker: str
    period: str
    data_points: list[dict]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_points(self) -> int:
        """Return number of data points in history"""
        return len(self.data_points)

    @property
    def period_return(self) -> Optional[float]:
        """
        Calculate total return over the period.

        Returns:
            Percentage return (eg: 15.5 for 15.5% gain)
        """
        if len(self.data_points) < 2:
            return None

        start_price = self.data_points[0].get("close")
        end_price = self.data_points[-1].get("close")

        if start_price and end_price:
            return ((end_price - start_price) / start_price) * 100

        return None

    def to_summary(self) -> str:
        """
        Format price history as a summary.

        Returns:
            String with period return and date range
        """
        return_str = f"{self.period_return:.2f}%" if self.period_return else "N/A"

        return (
            f"Period: {self.period} ({self.start_date} to {self.end_date})"
            f"Data Points: {self.total_points}\n"
            f"Period Return: {return_str}"
        )

@dataclass
class StockData:
    """
    Aggregated Stock data combining quote, info and history.

    This is the Primary data structure passed from the stock data tool
    to other parts of the agent.

    Attributes:
        ticker: Stock symbol
        quote: Current price and trading data
        company: Company information and fundamentals
        history: Historical price data
        error: Error message if data fetch failed
    """

    ticker: str
    quote: Optional[StockQuote] = None
    company: Optional[CompanyInfo] = None
    history: Optional[PriceHistory] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return if fetch was successfull"""
        return self.error is None and (self.quote is not None or self.company is not None)

    def to_context(self) -> str:
        """
        Format all stock data as context for the LLM.

        Returns:
            Comprehensive context string for memo generation
        """

        sections = [f"Sock Data for {self.ticker}\n"]

        if self.company:
            sections.append("\tCompany Overview")
            sections.append(self.company.to_context())

        if self.quote:
            sections.append("\tCurrent Quote")
            sections.append(self.quote.to_summary())

        if self.history:
            sections.append("\tPrice History")
            sections.append(self.history.to_summary())

        return "\n".join(sections)


# Stock Data Tool

class StockDataTool:
    """
    Tool for fetching stock matket data via Yfinance.

    This class wraps the yfinance library and provides:
        - Clean, types data models
        - Error handling and logging
        - Cachine (yet to implement)
        
    Example:
    -----------
        tool = StockDataTool()

        # Get everything about a stock
        data = tool.get_stock_data("AAPL")

        # or fetch specific pieces
        quote = tool.get_quote("AAPL")
        info = tool.get_company_info("AAPL")
        history = tool.get_price_history("AAPL", period="1y")
    """

    def __init__(self):
        """Initialize the stock data tool."""
        logger.info("StockDataTool Initialized!")

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Create a YFinance Ticker Object.

        Args:
            symbol: Stock ticker Symbol (eg: AAPL)

        Returns:
            yfinance.Ticker object
        """
        symbol = symbol.upper().strip()
        return yf.Ticker(symbol)


    def get_quote(self, symbol: str) -> StockQuote:
        """
        Fetch current stock quote.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Stockquote with current price and trading data.

        Raises:
            ValueError: if symbol is invalid or data unavailable
        """

        logger.info(f"Fetching quote for {symbol}")

        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            # Yfinance return empty dict for invalid ticker
            if not info or "regularMaketPrice" not in info:
                # Trying FastInfo as Fallback
                fast = ticker.fast_info
                if hasattr(fast, "last_price") and fast.last_price:
                    return StockQuote(
                        ticker=symbol.upper(),
                        current_price=fast.last_price,
                        previous_close=getattr(fast, "previous_close", None),
                        open_price=getattr(fast, "open", None),
                        day_high=getattr(fast, "day_high", None),
                        day_low=getattr(fast, "day_low", None),
                        fifty_two_week_high=getattr(fast, "year_high", None),
                        fifty_two_week_low=getattr(fast, "year_low", None),
                    )

                raise ValueError(f"No data found for symbol: {symbol}")

            return StockQuote(
                ticker=symbol.upper(),
                current_price=info.get("regularMarketPrice") or info.get("currentPrice", 0),
                open_price=info.get("regularMarketOpen") or info.get("open"),
                previous_close=info.get("regularMarketPreviousClose") or info.get("previousClose"),
                day_high=info.get("regularMarketDayHigh") or info.get("dayHigh"),
                day_low=info.get("regularMarketDayLow") or info.get("dayLow"),
                volume=info.get("regularMarketVolume") or info.get("volume"),
                average_volume=info.get("averageVolume"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            )

        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol.upper()}: {e}")
            raise ValueError(f"Failed to fetch quote for {symbol.upper()}: {e}")

    def get_company_info(self, symbol: str) -> CompanyInfo:
        """
        Fetch Company information and fundamentals.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CompanyInfo with company profile and metrics

        Raise:
            ValueError: If symbol is invalid or data unavailable
        """

        logger.info(f"Fetching company info for {symbol}")

        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or "shortName" not in info:
                raise ValueError(f"No Company info found for symbol: {symbol}")

            return CompanyInfo(
                ticker=symbol.upper(),
                name=info.get("shortName") or info.get("longName", symbol),
                sector=info.get("sector"),
                industry=info.get("industry"),
                description=info.get("longBusinessSummary"),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trainingPE"),
                forward_pe=info.get("forwardPE"),
                eps=info.get("trailingEps"),
                dividend_yield=info.get("dividendYield"),
                target_price=info.get("targetMeanPrice"),
                target_high=info.get("targetHighPrice"),
                target_low=info.get("targetLowPrice"),
                recommendation=info.get("recommendationKey"),
                website=info.get("website"),
                employees=info.get("fullTimeEmployees"),
            )
        except Exception as e:
            logger.error(f"Failed to fetch company info for {symbol.upper()}: {e}")
            raise ValueError(f"Failed to fetch company info for {symbol.upper()}: {e}")

    def get_price_history(
        self,
        symbol: str,
        period: str = "6mo",
    ) -> PriceHistory:
        """
        Fetch historical price data

        Args:
            symbol: Stock ticker symbol.
            period: Time period to fetch. Valid values:
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"

        Returns:
            PriceHistory with OHLCV data points

        Raises:
            ValueError: If symbol is invalid or no history available
        """

        logger.info(f"Fetching {period} price history for {symbol.upper()}")

        try:
            ticker = self._get_ticker(symbol)

            # fetch history as DataFrame
            df = ticker.history(period=period)

            if df.empty:
                raise ValueError(f"No price history found for {symbol.upper()}")

            data_points = []
            for date_idx, row in df.iterrows():
                data_points.append({
                    "date": date_idx.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                })

            return PriceHistory(
                ticker=symbol.upper(),
                period=period,
                data_points=data_points,
                start_date=data_points[0]["date"] if data_points else None,
                end_date=data_points[-1]["date"] if data_points else None,
            )

        except Exception as e:
            logger.error(f"Failed to fetch price history for {symbol.upper()}: {e}")
            raise ValueError(f"Failed to fetch price history for {symbol.upper()}: {e}")


    def get_stock_data(
        self,
        symbol: str,
        include_history: bool = True,
        history_period: str = "6mo",
    ) -> StockData:
        """
        Fetch comprehensive stock data (quote + company info + price history)

        This is the main method for getting all stock data at once.
        It handles partial failures gracefully - if one piece fails,
        others are still returned.

        Args:
            symbol: Stock ticker symbol
            include_history: Whether to fetch price history
            history_period: Period for price history

        Returns:
            StockData will all available data
        """

        logger.info(f"Fetching comprehensive data for {symbol.upper()}")

        quote = None
        company = None
        history = None
        errors = []

        # fetch quote
        try:
            quote = self.get_quote(symbol)
        except Exception as e:
            errors.append(f"Company: {e}")
            logger.warning(f"Could not fetch company info: {e}")

        # fetch company info
        try:
            company = self.get_company_info(symbol)
        except Exception as e:
            errors.append(f"Company: {e}")
            logger.warning(f"Could not fetch company info: {e}")

        # fetch price history if requested
        if include_history:
            try:
                history = self.get_price_history(symbol, period=history_period)
            except Exception as e:
                errors.append(f"History: {e}")
                logger.warning(f"Could not fetch history; {e}")

        error_msg = "; ".join(errors) if errors else None

        # only set error if fetch failed for all parts
        if not quote and not company:
            return StockData(
                ticker=symbol.upper(),
                error=error_msg or f"No data found for {symbol}",
            )

        return StockData(
            ticker=symbol.upper(),
            quote=quote,
            company=company,
            history=history,
            error=None,
        )


# Module level Wrapper functions

_default_tool: Optional[StockDataTool] = None

def _get_default_tool() -> StockDataTool:
    """Get the default StockDataTool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = StockDataTool()
    return _default_tool

def get_stock_quote(symbol: str) -> StockQuote:
    return _get_default_tool().get_quote(symbol)

def get_company_info(symbol: str) -> CompanyInfo:
    return _get_default_tool().get_company_info(symbol)

def get_price_history(symbol: str, period: str = "6mo") -> PriceHistory:
    return _get_default_tool().get_price_history(symbol, period=period)

def get_stock_data(symbol: str) -> StockData:
    return _get_default_tool().get_stock_data(symbol)


# Testing

if __name__ == "__main__":
    """
    Test the stock data tool from command line,

    Usage:
        python tools/stock_data_tool.py AAPL
        python tools/stock_data_tool.py MSFT GOOGL TSLA
    """
    
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Fetching data for: {symbol}")
        print(f"{'='*60}\n")

        data = get_stock_data(symbol)

        if data.success:
            print(data.to_context())
        else:
            print(f"Failed to fetch data: {data.error}")