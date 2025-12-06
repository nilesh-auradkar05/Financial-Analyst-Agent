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
from datetime import datetime, timezone
from typing import Optional, Literal
from pathlib import Path
from loguru import logger

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import (
    AgentState,
    AgentStep,
    NewsArticleState,
    StockDataState,
    FilingChunkState,
    SentimentResultState,
    Citation,
    create_initial_state,
    add_error_to_state,
    get_context_for_llm,
)

from tools.web_search_tool import WebSearchTool, search_company_news
from tools.stock_data_tool import StockDataTool, get_stock_data
from rag.vector_store import get_vector_store, search_filings
from models.sentiment import SentimentAnalyzer, analyze_sentiment_batch
from models.llm import get_llm, OllamaClient, FinancialPrompts

from configs.config import settings

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
        result = search_company_news(ticker, company_name)

        if not result.success:
            logger.warning(f"News search failed: {result.error}")
            return {
                "current_step": AgentStep.RESEARCH_NEWS.value,
                **add_error_to_state(state, "research_news", result.error or "Search failed"),
            }

        # Convert results to state format
        articles = [
            NewsArticleState(
                title=article.title,
                url=article.url,
                source=article.source,
                snippet=article.snippet,
                published_date=article.published_date,
                relevance_score=article.relevance_score,
            ).to_dict()
            for article in result.articles
        ]

        logger.info(f"[research_news] Found {len(articles)} articles")

        return {
            "news_articles": articles,
            "current_step": AgentStep.RESEARCH_NEWS.value,
        }

    except Exception as e:
        logger.error(f"[research_news] Error: {e}")
        return {
            "current_step": AgentStep.RESEARCH_NEWS.value,
            **add_error_to_state(state, "research_news", str(e)),
        }

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
        data = get_stock_data(ticker)

        if not data.success:
            logger.warning(f"Stock data fetch failed: {data.error}")
            return {
                "current_step": AgentStep.FETCH_STOCK.value,
                **add_error_to_state(state, "fetch_stock", data.error or "Fetch failed"),
            }

        # Build stock data state
        stock_state = StockDataState(
            ticker=ticker,
            company_name=data.company.name if data.company else ticker,
            current_price=data.quote.current_price if data.quote else None,
            price_change_percent=data.quote.day_change_percent if data.quote else None,
            market_cap=data.company.market_cap if data.company else None,
            pe_ratio=data.company.pe_ratio if data.company else None,
            fifty_two_week_high=data.quote.fifty_two_week_high if data.quote else None,
            fifty_two_week_low=data.quote.fifty_two_week_low if data.quote else None,
            target_price=data.company.target_price if data.company else None,
            company_description=data.company.description if data.company else None,
            sector=data.company.sector if data.company else None,
            industry=data.company.industry if data.company else None,
        )

        # Update company name if not set
        company_name = state.get("company_name") or stock_state.company_name

        logger.info(f"[fetch_stock] Got data for {company_name}")

        return {
            "stock_data": stock_state.to_dict(),
            "company_name": company_name,
            "current_step": AgentStep.FETCH_STOCK.value,
        }

    except Exception as e:
        logger.error(f"[fetch_stock] Error: {e}")
        return {
            "current_step": AgentStep.FETCH_STOCK.value,
            **add_error_to_state(state, "fetch_stock", str(e)),
        }

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
        queries = [
            f"{company_name} business description overview",
            f"{company_name} risk factors challenges",
            f"{company_name} revenue growth financial performance",
            f"{company_name} competition market position",
        ]

        all_chunks = []
        seen_texts = set()

        for query in queries:
            result = search_filings(query, ticker=ticker, n_results=3)

            for chunk in result.chunks:
                # Deduplicate by text
                text_hash = hash(chunk.text[:100])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_chunks.append(
                        FilingChunkState(
                            text=chunk.text,
                            section=chunk.metadata.get("section", "Unknown"),
                            filing_type=chunk.metadata.get("filing_type", "10-K"),
                            filing_date=chunk.metadata.get("filing_date"),
                            relevance_score=chunk.relevance_score,
                        ).to_dict()
                    )

        # Sort by relevance and take top chunks
        all_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        all_chunks = all_chunks[:10]

        logger.info(f"[retrieve_filings] Retrieved {len(all_chunks)} chunks")

        return {
            "filing_chunks": all_chunks,
            "current_step": AgentStep.RETRIEVE_FILINGS.value,
        }

    except Exception as e:
        logger.error(f"[retrieve_filings] Error: {e}")
        return {
            "current_step": AgentStep.RETRIEVE_FILINGS.value,
            **add_error_to_state(state, "retrieve_filings", str(e)),
        }

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
    articles = state.get("news_articles", [])

    logger.info(f"[analyze_sentiment] Analyzing {len(articles)} articles")

    if not articles:
        logger.warning("[analyze_sentiment] No articles to analyze")
        return {
            "sentiment_result": SentimentResultState(
                overall_sentiment="neutral",
            ).to_dict(),
            "current_step": AgentStep.ANALYZE_SENTIMENT.value,
        }

    try:
        # Extract snippets
        snippets = [article["snippet"] for article in articles]

        # Run batch sentiment analysis
        results = analyze_sentiment_batch(snippets)

        # Aggregate results
        positive_count = sum(1 for result in results if result.label == "positive")
        negative_count = sum(1 for result in results if result.label == "negative")
        neutral_count = sum(1 for result in results if result.label == "neutral")

        avg_positive = sum(result.positive_score for result in results) / len(results)
        avg_negative = sum(result.negative_score for result in results) / len(results)

        # Determine overall sentiment
        if positive_count > negative_count + 1:
            overall = "positive"
        elif negative_count > positive_count + 1:
            overall = "negative"
        else:
            overall = "neutral"

        # Build per-article breakdown
        article_sentiment = [
            {
                "title": articles[i]["title"],
                "sentiment": results[i].label,
                "confidence": results[i].confidence,
            }
            for i in range(len(results))
        ]

        sentiment_state = SentimentResultState(
            overall_sentiment=overall,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_positive_score=avg_positive,
            average_negative_score=avg_negative,
            article_sentiments=article_sentiment,
        )

        logger.info(f"[analyze_sentiment] Overall: {overall}")

        return {
            "sentiment_result": sentiment_state.to_dict(),
            "current_step": AgentStep.ANALYZE_SENTIMENT.value,
        }

    except Exception as e:
        logger.error(f"[analyze_sentiment] Error: {e}")
        return {
            "current_step": AgentStep.ANALYZE_SENTIMENT.value,
            **add_error_to_state(state, "analyze_sentiment", str(e)),
        }

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

        # Build prompt
        system_prompt = FinancialPrompts.ANALYST_SYSTEM

        user_prompt = f"""Generate a comprehensive investment memo for {company_name} ({ticker}).
        
        Based on the following data:

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

        async with OllamaClient() as client:
            response = await client.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.7,   # Slightly creative but factual
            )

        if not response.success:
            logger.error(f"[draft_memo] LLM failed: {response.error}")
            return {
                "current_step": AgentStep.DRAFT_MEMO.value,
                **add_error_to_state(state, "draft_memo", response.error or "LLM failed"),
            }

        memo = response.content

        # Build citations list
        citations = []
        citation_idx = 1

        # Add news citations
        for article in state.get("news_articles", [])[:5]:
            citations.append(Citation(
                index=citation_idx,
                source_type="news",
                title=article["title"],
                url=article["url"],
                date=article.get("published_date"),
            ).__dict__)
            citation_idx += 1

        # Add filing citations
        sections_cited = set()
        for chunk in state.get("filing_chunks", [])[:3]:
            section = chunk["section"]
            if section not in sections_cited:
                sections_cited.add(section)
                citations.append(Citation(
                    index=citation_idx,
                    source_type="filing",
                    title=f"10-K {section}",
                    date=chunk.get("filing_date"),
                ).__dict__)
                citation_idx += 1

        # Extract executive summary
        exec_summary = ""
        if "Executive Summary" in memo:
            parts = memo.split("Executive Summary")
            if len(parts) > 1:
                summary_section = parts[1].split("\n\n")[0]
                exec_summary = summary_section.strip().strip("#").strip()

        logger.info(f"[draft_memo] Drafted memo ({len(memo)} chars)")

        return {
            "investment_memo": memo,
            "citations": citations,
            "executive_summary": exec_summary,
            "current_step": AgentStep.DRAFT_MEMO.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"[draft_memo] Error: {e}")
        return {
            "current_step": AgentStep.DRAFT_MEMO.value,
            **add_error_to_state(state, "draft_memo", str(e)),
        }

# Graph Construction

def create_agent() -> StateGraph:
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
    start_time = datetime.now(timezone.utc)

    # Create agent
    agent = create_agent()

    # Create initial state
    initial_state = create_initial_state(ticker, company_name)

    # Execute the agent
    try:
        # Use ainvoke for async execution
        final_state = await agent.ainvoke(initial_state)

        # Calculate execution time
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        final_state["execution_time_ms"] = elapsed

        logger.info(f"Analysis complete for {ticker} in {elapsed:.0f}ms")

        return final_state

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Return state with error
        return {
            **initial_state,
            "current_step": AgentStep.ERROR.value,
            "errors": [{"step": "execution", "message": str(e)}],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

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
    print(f"Ticker: {result.get('ticker', 'N/A')}")
    print(f"Execution Time: {result.get('execution_time_ms', 0):.0f}ms")
    print(f"Errors: {len(result.get('errors', []))}")
    
    # Print errors if any
    errors = result.get("errors", [])
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - [{error['step']}] {error['message']}")
    
    # Print data gathered
    print(f"\nData Gathered:")
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