"""
Financial Analyst Agent Workflow Graph

This module defines the LangGraph workflow for the Financial Analyst Agent.
Design Pattern:
    - Conditional routing after each node for graceful degradation
    - Citation registry built *before* LLM call so indices are gounded

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
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from loguru import logger

from agents.state import (
    AgentState,
    AgentStep,
    add_error,
    create_initial_state,
    get_context_for_llm,
    get_data_availability,
    has_fatal_error,
)
from evaluation.grounding import evaluate_memo_grounding
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
        logger.info(f"[fetch_stock] {company_name} `[{ticker}]` Price: ${data.current_price}")

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
        seen_texts: set[str] = set()

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
    logger.info(f"[{ticker}] Analyzing sentiments on {len(articles)} articles")

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

def _build_citation_registry(state: AgentState, *, max_filing_chunks: int = 5) -> list[dict]:
    """
    Build an ordered list of citable sources from the state.

    Returns a list of dicts like:
        {"index": 1, "type": "news", "title": "....", "url": "...."}
    """
    registry: list[dict] = []
    idx = 1

    for article in state.get("news_articles", []):
        snippet = (article.get("snippet") or "").strip()
        if not snippet:
            continue

        registry.append({
            "index": idx,
            "type": "news",
            "title": article.get("title", "Untitled"),
            "text": snippet,
            "url": article.get("url", ""),
            "date": article.get("published_date"),
            "source": article.get("source", "Unknown"),
        })
        idx += 1

    for chunk in state.get("filing_chunks", [])[:max_filing_chunks]:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        section = chunk.get("section", "Unknown")
        filing_type = chunk.get("filing_type", "10-K")
        registry.append({
            "index": idx,
            "type": "sec_filing",
            "title": f"{section} - {filing_type}",
            "text": text,
            "url": chunk.get("source_url"),
            "date": chunk.get("filing_date"),
            "chunk_id": chunk.get("chunk_id"),
            "section": section,
            "filing_type": filing_type,
        })
        idx += 1

    return registry

def _format_registry_for_prompt(registry: list[dict], *, max_chars_per_item: int = 700) -> str:
    """Render the registry as a numbered list for the LLM prompt."""
    if not registry:
        return "\n## Available Sources\nNo numbered evidence sources are available. Avoid unsupported factual claims."

    lines = ["", "## Available Sources (use these citation numbers exactly)"]
    for entry in registry:
        meta_bits = [
            entry.get("type"),
            entry.get("source"),
            entry.get("date"),
        ]

        meta = " | ".join(bit for bit in meta_bits if bit)

        header = f"[{entry['index']}] {entry.get('title', 'Untitled')}"
        if meta:
            header += f" ({meta})"

        lines.append(header)
        lines.append((entry.get("text") or "")[:max_chars_per_item])
        lines.append("")
    return "\n".join(lines)

def _build_api_citations(registry: list[dict]) -> list[dict]:
    """Convert evidence registry into the citation metadata exposed by the API."""
    return [
        {
            "index": entry["index"],
            "type": entry.get("type", "unknown"),
            "title": entry.get("title", "Untitled"),
            "url": entry.get("url"),
            "date": entry.get("date"),
        }
        for entry in registry
    ]

def _extract_used_citations(memo: str) -> set[int]:
    """Pull citation indices like ``[1]``, ``[3]``, out of the memo text."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", memo)}

@traceable(name="draft_memo", run_type="llm", tags=["agent"])
async def draft_memo_node(state: AgentState) -> dict:
    """
    Draft the investment memo using the LLM.

    Compiles all gathered data into context and prompts
    Qwen3-VL to generate a comprehensive investment memo.

    Handles partial data: if sources are missing, the LLM is instructed
    not to fabricate information for those gaps.

    Args:
        state: Current agent state

    Returns:
        Partial state update with investment_memo and citations
    """

    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    logger.info(f"[draft_memo] Drafting memo for {ticker}")

    try:
        # 1. Data-availability check
        availability = get_data_availability(state)
        missing = [k.replace("has_", "") for k, v in availability.items() if not v]

        # 2. Build citation registry BEFORE LLM call
        registry = _build_citation_registry(state)
        api_citations = _build_api_citations(registry)

        # 3. Build context from accumulated data
        context = get_context_for_llm(state)
        registry_block = _format_registry_for_prompt(registry)
        logger.debug(f"[{ticker}] Context length: {len(context)} chars, "
                     f"Citations: {len(registry)}, Missing: {missing}")

        # 4. Build prompt with data-gap awareness
        disclaimer = ""
        if missing:
            disclaimer = (
                "\n\nIMPORTANT - these data sources were UNAVAILABLE: "
                + ", ".join(missing)
                + ". Do NOT fabricate information for them. "
                "State explicitly that the data was unavailable."
            )

        user_prompt = (
            f"Write a comprehensive investment memo for {company_name} ({ticker}).\n\n"
            f"{context}"
            f"{registry_block}\n\n"
            "Structure:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Company Overview\n"
            "3. Recent News & Market Sentiment\n"
            "4. Key Risk Factors (from SEC filings)\n"
            "5. Financial Highlights\n"
            "6. Investment Thesis (bullish and bearish cases)\n"
            "7. Conclusion with recommendation\n\n"
            "Use ONLY the numbered sources listed above for citations.\n"
            "Every non-trivial factual claim should cite one or more sources like [1] or [2][3].\n"
            "Do not invent citation numbers.\n"
            "Do not cite sources that are not listed.\n"
            "If support is weak or missing, explicitly state uncertainty."
            f"{disclaimer}"
        )

        llm = get_llm(temperature=0.7)
        logger.info(f"[{ticker}] Invoking LLM for memo generation...")
        response = await llm.ainvoke(
            [
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        memo = getattr(response, "content", "")

        used_citations = _extract_used_citations(memo)
        valid_indices = {entry["index"] for entry in registry}
        invalid_citations = sorted(used_citations - valid_indices)

        errors = list(state.get("errors", []))
        if invalid_citations:
            errors.append(
                {
                    "step": "draft_memo",
                    "message": f"Memo referenced non-existent_citations: {invalid_citations}",
                    "recoverable": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # 5. Extract executive summary
        exec_summary = ""
        if "Executive Summary" in memo:
            parts = memo.split("Executive Summary")
            if len(parts) > 1:
                summary = parts[1].split("\n\n")[0]
                exec_summary = summary.strip().strip("#").strip()[:500]

        logger.info(
            f"[{ticker}] Memo generated ({len(memo)} chars, used_citations={sorted(used_citations)})"
        )

        # 6. Verify citations
        used = _extract_used_citations(memo)
        valid_indices = {e["index"] for e in registry}
        orphan_indices = used - valid_indices
        if orphan_indices:
            logger.warning(
                f"[{ticker}] Memo references non-existent citations: "
                f"citations used: {sorted(used)}, missing data: {missing}"
            )

        return {
            "investment_memo": memo,
            "executive_summary": (
                exec_summary or f"Analysis completed for {company_name}"
            ),
            "citations": api_citations,
            "citation_evidence": registry,
            "errors": errors,
            "current_step": AgentStep.DRAFT_MEMO.value,
        }

    except Exception as e:
        logger.error(f"[{ticker}] Memo generation failed. Error: {e}")
        return add_error(state, "draft_memo", str(e), recoverable=False)

@traceable(name="verify_memo", run_type="chain", tags=["agent"])
async def verify_memo_node(state: AgentState) -> dict:
    """Run heuristic groundedness and citation coverage checks on the generated memo."""
    ticker = state["ticker"]
    memo = state.get("investment_memo", "")
    registry = state.get("citation_evidence", [])

    logger.info(f"[verify_memo] Verifying memo grounding for {ticker}")

    if not memo.strip():
        verification_result = {
            "passed": False,
            "total_claims": 0,
            "cited_claims": 0,
            "grounded_claims": 0,
            "citation_coverage_rate": 0.0,
            "grounded_claim_rate": 0.0,
            "claims": [],
            "orphan_citations": [],
        }
        errors = list(state.get("errors", []))
        errors.append(
            {
                "step": "verify_memo",
                "message": "No memo was available to verification.",
                "recoverable": True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        )
        return {
            "verification_result": verification_result,
            "errors": errors,
            "current_step": AgentStep.COMPLETE.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    used = _extract_used_citations(memo)
    valid_indices = {entry["index"] for entry in registry}
    orphan_indices = sorted(used - valid_indices)

    grounding = evaluate_memo_grounding(memo, registry)
    verification_result = grounding.to_dict()
    verification_result["orphan_citations"] = orphan_indices

    errors = list(state.get("errors", []))

    if orphan_indices:
        errors.append(
            {
                "step": "verify_memo",
                "message": f"Memo referenced non-existent citations: {orphan_indices}",
                "recoverable": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        verification_result["passed"] = False

    if not verification_result["passed"]:
        errors.append(
            {
                "step": "verify_memo",
                "message": (
                    "Memo verification failed "
                    f"(citation_coverage={verification_result['citation_coverage_rate']:.2f}), "
                    f"grounded_claim_rate={verification_result['grounded_claim_rate']:.2f}"
                ),
                "recoverable": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    logger.info(
        "[%s] Verification: passed=%s coverage=%.2f grounded=%.2f orphan_citations=%s",
        ticker,
        verification_result["passed"],
        verification_result["citation_coverage_rate"],
        verification_result["grounded_claim_rate"],
        orphan_indices,
    )

    return {
        "verification_result": verification_result,
        "errors": errors,
        "current_step": AgentStep.COMPLETE.value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

# Graph Construction

def _route_after_node(next_node: str):
    def router(state: AgentState) -> str:
        if has_fatal_error(state):
            logger.warning(
                "Fatal error - skipping to draft_memo"
                f"(world have gone to {next_node})"
            )
            return "draft_memo"
        return next_node
    return router

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
    workflow.add_conditional_edges(
        "research_news",
        _route_after_node("fetch_stock"),
        {"fetch_stock": "fetch_stock", "draft_memo": "draft_memo"},
    )
    workflow.add_conditional_edges(
        "fetch_stock",
        _route_after_node("retrieve_filings"),
        {"retrieve_filings": "retrieve_filings", "draft_memo": "draft_memo"},
    )
    workflow.add_conditional_edges(
        "retrieve_filings",
        _route_after_node("analyze_sentiment"),
        {"analyze_sentiment": "analyze_sentiment", "draft_memo": "draft_memo"},
    )
    workflow.add_conditional_edges(
        "analyze_sentiment",
        _route_after_node("draft_memo"),
        {"draft_memo": "draft_memo"},
    )
    workflow.add_edge("draft_memo", END)

    # Compile the graph
    agent = workflow.compile()
    logger.info("Agent Created with Conditional error routing Successfully")

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
