"""
Agent State Management Module

This module defines the state that flows through the LangGraph agent.
"""

import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Required, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ENUMS

class AgentStep(str, Enum):
    """
    Enumeration of agent workflow steps.

    Used for tracking progress and conditional routing.
    """
    RESEARCH_NEWS = "research_news"
    FETCH_STOCK = "fetch_stock"
    RETRIEVE_FILINGS = "retrieve_filings"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    DRAFT_MEMO = "draft_memo"
    COMPLETE = "complete"
    ERROR = "error"

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
    ticker: Required[str]
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
    started_at: str
    completed_at: str
    execution_time_ms: float

# STATE INITIALIZATION HELPERS

def create_initial_state(
    ticker: str,
    company_name: Optional[str] = None,
) -> AgentState:
    """Create initial state for agent run."""
    return AgentState(
        ticker=ticker.upper(),
        company_name=company_name or "",
        news_articles=[],
        stock_data={},
        filing_chunks=[],
        sentiment_result={},
        investment_memo="",
        executive_summary="",
        citations=[],
        current_step=AgentStep.RESEARCH_NEWS.value,
        errors=[],
        started_at=datetime.now(timezone.utc).isoformat(),
    )

def add_error(
    state: AgentState,
    step: str,
    message: str,
    recoverable: bool = True,
) -> dict:
    """Add error to state. Returns partial state update."""
    errors = state.get("errors", [])
    errors.append({
        "step": step,
        "message": message,
        "recoverable": recoverable,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "errors": errors,
        "current_step": AgentStep.ERROR.value if not recoverable else state.get("current_step"),
    }

def get_context_for_llm(state: AgentState) -> str:
    """Compile all data into context string for LLM."""
    parts = []
    citation_idx = 1

    # Stock data
    stock = state.get("stock_data", {})
    if stock:
        parts.append("## Stock Data")
        parts.append(f"Company: {stock.get('company_name', 'N/A')}")
        parts.append(f"Price: ${stock.get('current_price', 'N/A')}")
        parts.append(f"Market Cap: ${stock.get('market_cap', 'N/A'):,}" if stock.get('market_cap') else "")
        parts.append(f"P/E Ratio: {stock.get('pe_ratio', 'N/A')}")
        parts.append(f"Sector: {stock.get('sector', 'N/A')}")
        parts.append(f"52-Week Range: ${stock.get('fifty_two_week_low', 'N/A')} - ${stock.get('fifty_two_week_high', 'N/A')}")
        parts.append("")

    # News articles
    articles = state.get("news_articles", [])
    if articles:
        parts.append("## Recent News")
        for article in articles:
            parts.append(f"[{citation_idx}] {article.get('title', 'Untitled')}")
            parts.append(f"    Source: {article.get('source', 'Unknown')}")
            parts.append(f"    {article.get('snippet', '')[:200]}")
            parts.append("")
            citation_idx += 1

    # Sentiment
    sentiment = state.get("sentiment_result", {})
    if sentiment:
        parts.append("## Sentiment Analysis")
        parts.append(f"Overall: {sentiment.get('overall_sentiment', 'N/A')}")
        parts.append(f"Positive: {sentiment.get('positive_count', 0)} articles")
        parts.append(f"Negative: {sentiment.get('negative_count', 0)} articles")
        parts.append("")

    # Filing chunks
    chunks = state.get("filing_chunks", [])
    if chunks:
        parts.append("## SEC Filing Excerpts")
        for chunk in chunks[:5]:  # Limit to top 5
            parts.append(f"[{citation_idx}] {chunk.get('section', 'Unknown Section')}")
            parts.append(f"    {chunk.get('text', '')[:500]}")
            parts.append("")
            citation_idx += 1

    return "\n".join(parts)
