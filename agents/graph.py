"""
Financial Analyst Agent Workflow Graph

This module defines the LangGraph workflow for the Financial Analyst Agent.

Usage:
---------------
    from agent.graph import create_financial_analyst_graph

    # Create the agent
    agent = create_agent()

    # Run analysis
    result = await run_agent("AAPL")
    print(result["investment_memo"])
"""

import asyncio
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import (
    AgentState,
    AgentStep,
    add_error,
    create_initial_state,
    get_context_for_llm,
)
from models.llm import ANALYST_SYSTEM_PROMPT, get_llm
from models.sentiment import analyze_sentiment_batch
from observability.langsmith import get_tracer
from rag.vector_store import get_vector_store
from tools.stock_data_tool import get_stock_data
from tools.web_search_tool import search_company_news

# NODE FUNCTIONS

async def research_news_node(state: AgentState) -> dict:
    """
    Research recent news about the company.

    Uses Tavily to search for recent news articles,
    then stores them in state for sentiment analysis and citations.

    Args:
        state: Current agent state

    Returns:
        Partial state update with news_articles
    """
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)

    logger.info(f"[research_news] Searching news for {ticker}")

    try:
        # Search for news
        query = f"{company_name} {ticker} stock news"
        articles = await search_company_news(query, max_results=5)

        news_articles = [
            {
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "snippet": a.snippet,
                "published_date": a.published_date,
            }
            for a in articles
        ]

        logger.info(f"[{ticker}] Found {len(news_articles)} articles")

        return {
            "news_articles": news_articles,
            "current_step": AgentStep.RESEARCH_NEWS.value,
        }

    except Exception as e:
        logger.error(f"[{ticker}] News research failed. Error: {e}")
        return add_error(state, "research_news", str(e))

@traceable(name="fetch_stock", run_type="chain", tags=["agent"])
async def fetch_stock_node(state: AgentState) -> dict:
    """
    Fetch stock market data for the company.

    Uses YFinance to get current quote, company info, and price history.
    Updates company_name if not already set.

    Args:
        state: Current agent state

    Returns:
        Partial state update with stock_data
    """

    ticker = state["ticker"]

    logger.info(f"[fetch_stock] Fetching data for {ticker}")
    try:
        # Get stock data
        data = await get_stock_data(ticker)

        stock_data = {
            "ticker": data.ticker,
            "company_name": data.company_name,
            "current_price": data.current_price,
            "price_change_percent": data.price_change_percent,
            "market_cap": data.market_cap,
            "pe_ratio": data.pe_ratio,
            "fifty_two_week_high": data.fifty_two_week_high,
            "fifty_two_week_low": data.fifty_two_week_low,
            "volume": data.volume,
            "dividend_yield": data.dividend_yield,
            "sector": data.sector,
            "industry": data.industry,
            "recommendation": data.recommendation,
        }

        # Update company name if not set
        company_name = state.get("company_name") or data.company_name

        logger.info(f"[fetch_stock] Got data for {company_name}")
        logger.info(f"[{ticker}] Price: ${data.current_price}")

        return {
            "stock_data": stock_data,
            "company_name": company_name,
            "current_step": AgentStep.FETCH_STOCK.value,
        }

    except Exception as e:
        logger.error(f"[{ticker}] Stock Fetch Failed. Error: {e}")
        return add_error(state, "fetch_stock", str(e))

@traceable(name="retrieve_filings", run_type="retriever", tags=["agent"])
async def retrieve_sec_filings_node(state: AgentState) -> dict:
    """
    Retrieve relevant SEC filing chunks from vector store.

    Queries ChromaDB for chunks matching the company and key topics
    (risk factors, business description, etc.).

    Args:
        state: Current agent state

    Returns:
        Partial state update with filing_chunks
    """
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)

    logger.info(f"[retrieve_filings] Searching filings for {ticker}")

    try:
        store = get_vector_store()

        # Multiple queries for comprehensive retrieval
        queries = [
            f"{company_name} business description overview",
            f"{company_name} risk factors challenges",
            f"{company_name} financial performance revenue",
            f"{company_name} competition market position",
        ]

        all_chunks = []
        seen_texts = set()

        for query in queries:
            result = store.search_by_ticker(query, ticker=ticker, n_results=3)

            for chunk in result.chunks:
                # Deduplicate by text
                text_hash = hashlib.md5(chunk.text[:100].encode("utf-8")).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_chunks.append(chunk)

        # Sort by relevance and take top chunks
        all_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        top_chunks = all_chunks[:10]

        filing_chunks = [
            {
                "text": c.text,
                "section": c.section or "Unknown",
                "filing_type": c.metadata.get("filing_type", "10-K"),
                "filing_date": c.filing_date,
                "relevance_score": c.relevance_score,
            }
            for c in top_chunks
        ]

        logger.info(f"[retrieve_filings] Retrieved {len(filing_chunks)} chunks")

        return {
            "filing_chunks": filing_chunks,
            "current_step": AgentStep.RETRIEVE_FILINGS.value,
        }

    except Exception as e:
        logger.error(f"[{ticker}] Filing retrieval failed. Error: {e}")
        return add_error(state, "retrieve_filings", str(e))

@traceable(name="analyze_sentiment", run_type="chain", tags=["agent"])
async def analyze_sentiment_node(state: AgentState) -> dict:
    """
    Analyze sentiment of news articles using FinBERT.

    Runs batch sentiment analysis on article snippets,
    then aggregates results for the overall sentiment.

    Args:
        state: Current agent state

    Returns:
        Partial state update with sentiment_result
    """
    ticker = state["ticker"]
    articles = state.get("news_articles", [])

    logger.info(f"[{ticker}] Analyzing articles sentiment....")

    if not articles:
        return {
            "sentiment_result": {"overall_sentiment": "neutral"},
            "current_step": AgentStep.ANALYZE_SENTIMENT.value,
        }

    try:
        texts = [a["snippet"] for a in articles if a.get("snippet")]
        results = analyze_sentiment_batch(texts)

        positive = sum(1 for r in results if r.label == "positive")
        negative = sum(1 for r in results if r.label == "negative")
        neutral = sum(1 for r in results if r.label == "neutral")

        # Determine overall sentiment
        if positive > negative:
            overall = "positive"
        elif negative > positive:
            overall = "negative"
        else:
            overall = "neutral"

        sentiment_result = {
            "overall_sentiment": overall,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
        }

        logger.info(f"[{ticker}] Sentiment: {overall} ({positive}+/{negative}-)")

        return {
            "sentiment_result": sentiment_result,
            "current_step": AgentStep.ANALYZE_SENTIMENT.value,
        }

    except Exception as e:
        logger.error(f"[{ticker}] Sentiment analysis failed. Error: {e}")
        return add_error(state, "analyze_sentiment", str(e))

@traceable(name="draft_memo", run_type="llm", tags=["agent"])
async def draft_memo_node(state: AgentState) -> dict:
    """
    Draft the investment memo using the LLM.

    Compiles all gathered data into context and prompts
    Qwen3-VL to generate a comprehensive investment memo.

    Args:
        state: Current agent state

    Returns:
        Partial state update with investment_memo and citations
    """

    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)

    logger.info(f"[draft_memo] Drafting memo for {ticker}")

    try:
        # Build context from accumulated data
        context = get_context_for_llm(state)
        logger.debug(f"[{ticker}] Context length: {len(context)} chars")

        user_prompt = f"""Based on the following research, write a comprehensive investment memo for {company_name} ({ticker}).

        {context}

        Your memo should include:
            1. Executive Summary (2-3 sentences)
            2. Company Overview
            3. Recent News & Market Sentiment
            4. Key Risk Factors (from SEC filings)
            5. Financial Highlights
            6. Investment Thesis (bullish and bearish cases)
            7. Conclusion with recommendation

        Format the memo in markdown. Include citation numbers [1], [2], etc. for claims.
        Be objective and data-driven. Acknowledge uncertainties."""

        llm = get_llm(temperature=0.7)
        logger.info(f"[{ticker}] Invoking LLM for memo generation...")
        response = await llm.ainvoke(
            [
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        memo = getattr(response, "content", "")

        # Extract executive summary
        exec_summary = ""
        if "Executive Summary" in memo:
            parts = memo.split("Executive Summary")
            if len(parts) > 1:
                summary = parts[1].split("\n\n")[0]
                exec_summary = summary.strip().strip("#").strip()[:500]

        # Build citations list
        citations = []
        citation_idx = 1

        # Add news citations
        for article in state.get("news_articles", []):
            citations.append({
                "index": citation_idx,
                "type": "news",
                "title": article.get("title", "Untitled"),
                "url": article.get("url", ""),
            })
            citation_idx += 1

        # Add filing citations
        for chunk in state.get("filing_chunks", [])[:3]:
            section = chunk.get("section", "")
            filing_type = chunk.get("filing_type", "10-K")
            citations.append({
                "index": citation_idx,
                "type": "sec_filing",
                "title": f"{section} - {filing_type}",
            })
            citation_idx += 1

        logger.info(f"[{ticker}] Memo generated ({len(memo)} chars)")

        return {
            "investment_memo": memo,
            "executive_summary": exec_summary or f"Analysis complete for {company_name}",
            "citations": citations,
            "current_step": AgentStep.COMPLETE.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"[{ticker}] Memo generation failed. Error: {e}")
        return add_error(state, "draft_memo", str(e), recoverable=False)

# Graph Construction

def create_agent() -> Runnable[AgentState, AgentState]:
    """
    Create the LangGraph agent workflow.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating Financial Analyst Agent")

    # Create graph with our state schema
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("research_news", research_news_node)
    workflow.add_node("fetch_stock", fetch_stock_node)
    workflow.add_node("retrieve_filings", retrieve_sec_filings_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("draft_memo", draft_memo_node)

    # Define edges
    workflow.add_edge(START, "research_news")
    workflow.add_edge("research_news", "fetch_stock")
    workflow.add_edge("fetch_stock", "retrieve_filings")
    workflow.add_edge("retrieve_filings", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "draft_memo")
    workflow.add_edge("draft_memo", END)

    # Compile the graph
    agent = workflow.compile()

    logger.info("Agent Created Successfully")

    return agent

# AGENT EXECUTION

@traceable(name="run_financial_analysis", tags=['agent', "main"])
async def run_agent(
    ticker: str,
    company_name: Optional[str] = None,
) -> AgentState:
    """
    Run the financial analyst agent for a given ticker.

    This is the main entry point for agent execution.

    Args:
        ticker: Stock ticker symbol
        company_name: Company name (optional)

    Returns:
        Final agent state with all results

    Example:
        result = await run_agent("AAPL")
        print(result["investment_memo"])
    """

    logger.info(f"Starting analysis for {ticker}")

    # Create initial state
    state = create_initial_state(ticker, company_name)

    # Create agent
    agent = create_agent()

    # Get tracer for LangGraph
    tracer = get_tracer()
    config: RunnableConfig | None = {"callbacks": [tracer]} if tracer else {}

    # Excecute
    start_time = datetime.now(timezone.utc)
    final_state = await agent.ainvoke(state, config=config)

    # Calculate execution time
    exec_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    final_state["execution_time_ms"] = exec_time

    logger.info(f"Analysis complete for {ticker} in ({exec_time:.0f}ms)")

    return final_state

def run_agent_sync(
    ticker: str,
    company_name: Optional[str] = None,
) -> AgentState:
    """
    Run the agent synchronously. (Wrapper function)

    Args:
        ticker: Stock ticker symbol
        company_name: Company name

    Returns:
        Final agent state
    """
    return asyncio.run(run_agent(ticker, company_name))

# Testing
async def _main():
    """Test the agent."""
    import sys

    print(f"\n{'='*60}")
    print("Financial Analyst Agent - Testing")
    print(f"{'='*60}\n")

    # Get ticker from args
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"Analyzing: {ticker}")
    print("This may take a few minutes...\n")

    # Run the agent
    result = await run_agent(ticker)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    print(f"Company: {result.get('company_name', 'N/A')}")
    print(f"Status: {result.get('current_step')}")
    print(f"Execution Time: {result.get('execution_time_ms', 0):.0f}ms")
    print(f"Errors: {len(result.get('errors', []))}")

    # Print errors if any
    errors = result.get("errors", [])
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - [{error['step']}] {error['message']}")

    # Print data gathered
    print("\nData Gathered:")
    print(f"  - News Articles: {len(result.get('news_articles', []))}")
    print(f"  - Filing Chunks: {len(result.get('filing_chunks', []))}")
    print(f"  - Stock Data: {'Yes' if result.get('stock_data') else 'No'}")
    print(f"  - Sentiment: {result.get('sentiment_result', {}).get('overall_sentiment', 'N/A')}")

    # Print executive summary
    exec_summary = result.get("executive_summary", "")
    if exec_summary:
        print(f"\n{'─'*60}")
        print("EXECUTIVE SUMMARY")
        print(f"{'─'*60}")
        print(exec_summary[:500] + "..." if len(exec_summary) > 500 else exec_summary)

    # Print memo preview
    memo = result.get("investment_memo", "")
    if memo:
        print(f"\n{'─'*60}")
        print("INVESTMENT MEMO (Preview)")
        print(f"{'─'*60}")
        print(memo[:1000] + "..." if len(memo) > 1000 else memo)

    # Print citations
    citations = result.get("citations", [])
    if citations:
        print(f"\n{'─'*60}")
        print("CITATIONS")
        print(f"{'─'*60}")
        for cite in citations[:5]:
            print(f"  [{cite['index']}] ({cite['source_type']}) {cite['title']}")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    asyncio.run(_main())
