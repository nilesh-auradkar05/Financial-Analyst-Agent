"""
Agent State Management Module

This module defines the state that flows through the LangGraph agent.
"""

from nntplib import ArticleInfo
from typing import TypedDict, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import operator

from langchain_core.messages import BaseMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ENUMS

class AgentStep(str, Enum):
    """
    Enumeration of agent workflow steps.

    Used for tracking progress and conditional routing.
    """

    INITIALIZE = "initialize"
    RESEARCH_NEWS = "research_news"
    FETCH_STOCK = "fetch_stock"
    RETRIEVE_FILINGS = "retrieve_filings"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    DRAFT_MEMO = "draft_memo"
    COMPLETE = "complete"
    ERROR = "error"

class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

# DATA Classes for State Components

@dataclass
class NewsArticleState:
    """
    News article data stored in agent state.
    
    Abstracted version of NewsArticle from web_search_tool,
    containing only what is needed for the agent workflow.
    
    Attributes:
        title: Article headline
        url: Link for citation
        source: Publisher name
        snippet: Text content for analysis
        published_date: Publication date
        relevance_score: Search relevance (0-1)
    """
    title: str
    url: str
    source: str
    snippet: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0

    def to_citation(self, index: int) -> str:
        """Format as citation string"""
        date_str = f", {self.published_date}" if self.published_date else ""
        return f"[{index}] {self.title} - {self.source}{date_str}\n    {self.url}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            "published_date": self.published_date,
            "relevance_score": self.relevance_score,
        }

@dataclass
class StockDataState:
    """
    Stock market data stored in agent state.
    
    Contains key metrics needed for the investment memo.
    
    Attributes:
        current_price: Latest stock price
        price_change_percent: Daily change percentage
        market_cap: Market capitalization
        pe_ratio: Price-to-earnings ratio
        fifty_two_week_high: 52-week high price
        fifty_two_week_low: 52-week low price
        target_price: Analyst target price
        company_description: Business summary
        sector: Business sector
        industry: Specific industry
    """
    
    ticker: str
    company_name: str
    current_price: Optional[float] = None
    price_change_percent: Optional[float] = None
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    target_price: Optional[float] = None
    company_description: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    def to_summary(self) -> str:
        """Format as summary for memo."""
        lines = [
            f"Company: {self.company_name} ({self.ticker})",
            f"Sector: {self.sector or 'N/A'} | Industry: {self.industry or 'N/A'}",
            f"Current Price: ${self.current_price:.2f}" if self.current_price else "Price: N/A",
        ]

        if self.market_cap:
            if self.market_cap >= 1_000_000_000_000:
                cap_str = f"${self.market_cap / 1_000_000_000_000:.2f}T"
            elif self.market_cap >= 1_000_000_000:
                cap_str = f"${self.market_cap / 1_000_000_000:.2f}B"
            else:
                cap_str = f"${self.market_cap / 1_000_000:.2f}M"
            lines.append(f"Market Cap: {cap_str}")

        if self.pe_ratio:
            lines.append(f"P/E Ratio: {self.pe_ratio:.2f}")

        if self.target_price:
            lines.append(f"Analyst Target: ${self.target_price:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "current_price": self.current_price,
            "price_change_percent": self.price_change_percent,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "fifty_two_week_high": self.fifty_two_week_high,
            "fifty_two_week_low": self.fifty_two_week_low,
            "target_price": self.target_price,
            "sector": self.sector,
            "industry": self.industry,
        }

@dataclass
class FilingChunkState:
    """
    SEC filing chunk stored in agent state.
    
    Retrieved from vector store for RAG context.
    
    Attributes:
        text: Chunk content
        section: Section name (e.g., "Risk Factors")
        filing_type: Filing type (e.g., "10-K")
        filing_date: Filing date
        relevance_score: Search relevance (0-1)
    """
    
    text: str
    section: str
    filing_type: str = "10-K"
    filing_date: Optional[str] = None
    relevance_score: float = 0.0

    def to_context(self) -> str:
        """Format as context for LLM."""
        return f"[{self.section} - {self.filing_type}]\n{self.text}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "section": self.section,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "relevance_score": self.relevance_score,
        }

@dataclass
class SentimentResultState:
    """
    Sentiment analysis results stored in agent state.
    
    Aggregated sentiment from news articles.
    
    Attributes:
        overall_sentiment: Overall classification
        positive_count: Number of positive articles
        negative_count: Number of negative articles
        neutral_count: Number of neutral articles
        average_score: Average sentiment score
        article_sentiments: Per-article sentiment breakdown
    """
    
    overall_sentiment: str
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_positive_score: float = 0.0
    average_negative_score: float = 0.0
    article_sentiments: list[dict] = field(default_factory=list)

    def to_summary(self) -> str:
        """Format as summary for memo."""
        total = self.positive_count + self.negative_count + self.neutral_count

        if total == 0:
            return "No sentiment data available."

        return f"""Overall Sentiment: {self.overall_sentiment.upper()}
        Distribution: {self.positive_count} positive, {self.negative_count} negative, {self.neutral_count} neutral
        Positive Score: {self.average_positive_score:.1%}
        Negative Score: {self.average_negative_score:.1%}
        """

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall_sentiment": self.overall_sentiment,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "average_positive_score": self.average_positive_score,
            "average_negative_score": self.average_negative_score,
            "article_sentiments": self.article_sentiments,
        }

@dataclass
class Citation:
    """
    Citation for the investment memo.
    
    Tracks the source of each claim in the memo.
    
    Attributes:
        index: Citation number [1], [2], etc.
        source_type: Type of source (news, filing, stock_data)
        title: Source title
        url: Link to source (if available)
        date: Source date
        excerpt: Relevant excerpt
    """
    
    index: int
    source_type: str  # "news", "filing", "stock_data"
    title: str
    url: Optional[str] = None
    date: Optional[str] = None
    excerpt: Optional[str] = None

    def to_string(self) -> str:
        """Format as citation string."""
        parts = [f"[{self.index}]", f"(self.source_type)", self.title]

        if self.date:
            parts.append(f"- {self.date}")

        result = " ".join(parts)

        if self.url:
            result += f"\n    {self.url}"

        return result

@dataclass
class AgentError:
    """
    Error that occurred during agent execution.
    
    Stored in state for graceful degradation.
    
    Attributes:
        step: Which step failed
        message: Error message
        timestamp: When error occurred
        recoverable: Can the agent continue?
    """

    step: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    recoverable: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "step": self.step,
            "message": self.message,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
        }

# MAIN AGENT STATE

class AgentState(TypedDict, total=False):
    """
    Main state definition for the Financial Analyst Agent.
    
    Using total=False makes all fields optional, allowing partial state
    updates from each node.
    
    State Categories:
    -----------------
    1. INPUT: What the user provides (ticker)
    2. ACCUMULATED: Data gathered during workflow
    3. OUTPUT: Final results (memo, citations)
    4. METADATA: Tracking and debugging info
    
    Usage:
    ------
        # Initial state
        state = AgentState(
            ticker="AAPL",
            company_name="Apple Inc",
        )
        
        # Node updates state (returns partial dict)
        def research_node(state: AgentState) -> dict:
            articles = search_news(state["ticker"])
            return {"news_articles": articles}
        
        # LangGraph merges the update into state
    """

    # Input Fields
    ticker: str
    company_name: str

    # Accumulated Data Fields
    news_articles: list[dict]
    stock_data: dict
    filing_chunks: list[dict]
    sentiment_result: dict

    # Output Fields
    investment_memo: str
    citations: list[dict]
    executive_summary: str

    # Metadata Fields
    current_step: str
    errors: list[dict]
    message: list[BaseMessage]
    started_at: str
    completed_at: str
    execution_time_ms: float

# STATE INITIALIZATION HELPERS

def create_initial_state(
    ticker: str,
    company_name: Optional[str] = None,
) -> AgentState:
    """
    Create initial state for agent execution.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name (optional, can be fetched later)
        
    Returns:
        Initialized AgentState
    """
    return AgentState(
        ticker=ticker.upper(),
        company_name=company_name or "",
        news_articles=[],
        stock_data={},
        filing_chunks=[],
        sentiment_result={},
        investment_memo="",
        citations=[],
        executive_summary="",
        current_step=AgentStep.INITIALIZE.value,
        errors=[],
        messages=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at="",
        execution_time_ms=0.0,
    )

def add_error_to_state(
    state: AgentState,
    step: str,
    message: str,
    recoverable: bool = True,
) -> dict:
    """
    Add an error to the state.
    
    Returns a partial state update.
    
    Args:
        state: Current state
        step: Which step failed
        message: Error message
        recoverable: Can execution continue?
        
    Returns:
        Partial state dict with updated errors
    """
    error = AgentError(
        step=step,
        message=message,
        recoverable=recoverable,
    )

    current_errors = state.get("errors", [])

    return {
        "errors": current_errors + [error.to_dict()],
    }

def get_context_for_llm(state: AgentState) -> str:
    """
    Build context string for LLM from accumulated state.
    
    This compiles all gathered data into a format suitable
    for the memo generation step.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted context string
    """
    sections = []

    # Company info
    ticker = state.get("ticker", "")
    company = state.get("company_name", "")
    sections.append(f"# Analysis for {company} ({ticker})\n")

    # Stock data
    stock_data = state.get("stock_data", {})
    if stock_data:
        stock_state = StockDataState(**stock_data)
        sections.append("## Stock Data")
        sections.append(stock_state.to_summary())
        sections.append("")

    # News articles
    articles = state.get("news_articles", [])
    if articles:
        sections.append("## Recent News")
        for i, article_dict in enumerate(articles[:5], 1):
            article = NewsArticleState(**article_dict)
            sections.append(f"{i}. {article.title} ({article.source})")
            sections.append(f"    {article.snippet[:200]}...")
        sections.append("")

    # Sentiment
    sentiment = state.get("sentiment_result", {})
    if sentiment:
        sent_state = SentimentResultState(**sentiment)
        sections.append("## Sentiment Analysis")
        sections.append(sent_state.to_summary())
        sections.append("")

    # Filing Chunks
    chunks = state.get("filing_chunks", [])
    if chunks:
        sections.append("## SEC Filing Experts")
        for chunk_dict in chunks[:5]:
            chunk = FilingChunkState(**chunk_dict)
            sections.append(f"### {chunk.section}")
            sections.append(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
            sections.append("")

    return "\n".join(sections)

# STATE VALIDATION

def validate_state(state: AgentState) -> list[str]:
    """
    Validate agent state and return list of issues.
    
    Args:
        state: State to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check required input
    if not state.get("ticker"):
        issues.append("Missing required field: ticker")

    if not state.get("news_articles") and state.get("current_step") in [
        AgentStep.RESEARCH_NEWS.value,
        AgentStep.DRAFT_MEMO.value,
    ]:
        issues.append("No news articles available for analysis")

    if not state.get("filing_chunks") and state.get("current_step") == AgentStep.DRAFT_MEMO.value:
        issues.append("No SEC filing data available")
    return issues

# Module EXPORTS

__all__ = [
    # Main state
    "AgentState",
    "AgentStep",

    # State Components
    "NewsArticleState",
    "StockDataState",
    "FilingChunkState",
    "SentimentResultState",
    "Citation",
    "AgentError",

    # Helpers
    "create_initial_state",
    "add_error_to_state",
    "get_context_for_llm",
    "validate_state",
]