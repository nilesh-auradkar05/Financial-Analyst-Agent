"""
FastAPI Service

This module implements the FastAPI service for the financial analyst agent system.

Features:
    - Sync analysis endpoint
    - Async job-based analysis endpoint.
    - Health check for ollama server.
    - Filing ingestion endpoint.
    - Proper error handling and logging
    - CORS support for web clients.

Endpoints:
---------------

POST /analyze            - Run analysis (sync, blocks until complete)
POST /analyze/async      - Start analysis job
GET /jobs/{job_id}       - Get job status and results
POST /ingest             - Ingest SEC filings for a company
GET /health              - Health check
GET /stats               - System statistics

Usage:
-----------
    # start server
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # or
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
"""

import asyncio
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local Imports
from api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    IngestionRequest,
    IngestionResponse,
    JobStatusResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    ErrorDetail,
    NewsArticleResponse,
    StockDataResponse,
    SentimentResponse,
    CitationResponse,
    JobStatus,
)

from agents.graph import run_agent
from agents.state import AgentState
from rag.ingestion import ingest_10k_for_ticker
from rag.vector_store import get_vector_store
from models.llm import check_ollama_health
from configs.config import settings, validate_settings

# Job Storage
# Current implementation is in-memory storage

class JobStorage:
    """In-memory job storage.
    
    Stores job status and results for async analysis.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._results: dict[str, AgentState] = {}

    def create_job(self, ticker: str) -> str:
        """Create a new analysis job.
        
        Args:
            ticker: Stock ticker symbol

        Returns:
            Job ID
        """

        job_id = str(uuid.uuid4())[:8]

        self._jobs[job_id] = {
            "job_id": job_id,
            "ticker": ticker.upper(),
            "status": JobStatus.PENDING,
            "current_step": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }

        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        current_step: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update job status."""

        if job_id not in self._jobs:
            return
        
        if status:
            self._jobs[job_id]["status"] = status
        if current_step:
            self._jobs[job_id]["current_step"] = current_step
        if error:
            self._jobs[job_id]["error"] = error

        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details"""
        return self._jobs.get(job_id)

    def store_result(self, job_id: str, result: AgentState):
        """Store job results."""
        self._results[job_id] = result

    def get_result(self, job_id: str) -> Optional[AgentState]:
        """Get job results"""
        return self._results.get(job_id)

    def get_stats(self) -> dict:
        """Get job statistics"""
        completed = sum(
            1 for job in self._jobs.values()
            if job["status"] == JobStatus.COMPLETED
        )

        failed = sum(
            1 for job in self._jobs.values()
            if job["status"] == JobStatus.FAILED
        )

        return {
            "total_jobs": len(self._jobs),
            "completed_jobs": completed,
            "failed_jobs": failed,
            "pending_jobs": len(self._jobs) - completed - failed,
        }

job_store = JobStorage()

# Track server start time
SERVER_START_TIME = datetime.now(timezone.utc)

# Application State

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Applicaiton lifespan manager.
    
    Runs setup on startup and cleanup on shutdown.
    """

    logger.info("Starting Financial Analyst System...")

    # validate settings
    warnings = validate_settings()
    for warning in warnings:
        logger.warning(warning)

    # initialize vector store
    try:
        store = get_vector_store()
        logger.info(f"Vector store ready: {store.count} documents")
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")

    # Check ollama endpoint health
    try:
        ollama_healthy = await check_ollama_health()
        if ollama_healthy:
            logger.info("Ollama connection successful")
        else:
            logger.warning("Ollama connection failed. LLM services will fail.")
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")

    logger.info("Application ready")

    yield

    # Shutdown
    logger.info("Shutting down Financial Analyst System API...")

# FastAPI application
app = FastAPI(
    title="Financial Analyst System API",
    description="""
    AI-powered Financial Analyst Agent API.
    
    Analyzes companies by:
    - Searching recent news
    - Fetching stock market data
    - Analyzing SEC filings
    - Running sentiment analysis
    - Generating investment memos
    
    ## Quick Start
    
    1. **Ingest filings** (one-time per company):
    ```
       POST /ingest {"ticker": "AAPL"}
    ```bash
    
    2. **Run analysis**:
    ```bash
       POST /analyze {"ticker": "AAPL"}
    ```bash
    
    3. **Get results** (if using async):
    ```
       GET /jobs/{job_id}
    ```bash
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.exception(f"Unexpected error: {exc}")

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server error",
            detail=str(exc) if settings.debug else None,
            status_code=500,
        ).model_dump(),
    )

# Helper Functions
def state_to_response(state: AgentState, job_id: Optional[str] = None) -> AnalysisResponse:
    """Convert agent state to analysis response.
    
    Args:
        state: Agent state
        job_id: Job ID (optional)

    Returns:
        Analysis response
    """
    stock_data = None
    sd = state.get("stock_data")
    if sd:

        # Format market cap
        market_cap_formatted = None
        if sd.get("market_cap"):
            market_cap = sd["market_cap"]
            if market_cap >= 1_000_000_000_000:
                market_cap_formatted = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                market_cap_formatted = f"${market_cap / 1_000_000_000:.2f}B"
            else:
                market_cap_formatted = f"${market_cap / 1_000_000:.2f}M"


        stock_data = StockDataResponse(
            ticker=sd.get("ticker", ""),
            company_name=sd.get("company_name", ""),
            current_price=sd.get("current_price"),
            price_change_percent=sd.get("price_change_percent"),
            market_cap=sd.get("market_cap"),
            market_cap_formatted=market_cap_formatted,
            pe_ratio=sd.get("pe_ratio"),
            fifty_two_week_high=sd.get("fifty_two_week_high"),
            fifty_two_week_low=sd.get("fifty_two_week_low"),
            target_price=sd.get("target_price"),
            sector=sd.get("sector"),
            industry=sd.get("industry"),
        )

    # Convert sentiment
    sentiment = None
    sr = state.get("sentiment_result")
    if sr:
        sentiment = SentimentResponse(
            overall_sentiment=sr.get("overall_sentiment", "neutral"),
            positive_count=sr.get("positive_count", 0),
            negative_count=sr.get("negative_count", 0),
            neutral_count=sr.get("neutral_count", 0),
            average_positive_score=sr.get("average_positive_score", 0.0),
            average_negative_score=sr.get("average_negative_score", 0.0),
        )

    # convert news articles
    news_articles = []
    sentiment_results = state.get("sentiment_result", {}).get("article_sentiments", [])

    for i, article in enumerate(state.get("news_articles", [])):
        article_sentiment = sentiment_results[i] if i < len(sentiment_results) else {}

        news_articles.append(NewsArticleResponse(
            title=article.get("title", ""),
            url=article.get("url", ""),
            source=article.get("source", ""),
            snippet=article.get("snippet", ""),
            published_date=article.get("published_date", ""),
            relevance_score=article.get("relevance_score", 0.0),
            sentiment=article_sentiment.get("sentiment"),
            sentiment_confidence=article_sentiment.get("confidence", 0.0),
        ))

    # convert citations
    citations = [
        CitationResponse(
            index=citation.get("index", 0),
            source_type=citation.get("source_type", ""),
            title=citation.get("title", ""),
            url=citation.get("url"),
            date=citation.get("date"),
        )
        for citation in state.get("citations", [])
    ]

    # Convert errors
    errors = [
        ErrorDetail(
            step=error.get("step", ""),
            message=error.get("message", ""),
            timestamp=error.get("timestamp", ""),
            recoverable=error.get("recoverable", True),
        )
        for error in state.get("errors", [])
    ]

    # Determine status
    if state.get("investment_memo"):
        status = JobStatus.COMPLETED
    elif state.get("errors"):
        status = JobStatus.FAILED if not any(error.recoverable for error in errors) else JobStatus.COMPLETED
    else:
        status = JobStatus.FAILED

    return AnalysisResponse(
        job_id=job_id,
        ticker=state.get("ticker", ""),
        company_name=state.get("company_name", ""),
        status=status,
        executive_summary=state.get("executive_summary"),
        investment_memo=state.get("investment_memo"),
        stock_data=stock_data,
        sentiment=sentiment,
        news_articles=news_articles,
        citations=citations,
        errors=errors,
        started_at=state.get("statrted_at"),
        completed_at=state.get("completed_at"),
        execution_time_ms=state.get("execution_time_ms"),
    )

async def run_analysis_job(job_id: str, request: AnalysisRequest):
    """
    Background task for running analysis.

    Args:
        - job_id: Job identifier
        - request: Analysis request
    """
    logger.info(f"[Job {job_id}] Starting analysis for {request.ticker}")

    try:
        job_store.update_job(job_id, status=JobStatus.RUNNING)

        # Run the agent
        result = await run_agent(
            ticker=request.ticker,
            company_name=request.company_name,
        )

        job_store.store_result(job_id, result)

        # Update job status
        if result.get("investment_memo"):
            job_store.update_job(job_id, status=JobStatus.COMPLETED)
            logger.info(f"[Job {job_id}] Analysis completed successfully")
        else:
            error_msg = "; ".join(
                error.get("message", "Unknown error")
                for error in result.get("errors", [])
            )
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=error_msg or "Analysis produced no results",
            )
            logger.error(f"[Job {job_id}] Analysis failed: {error_msg}")

    except Exception as e:
        logger.exception(f"[Job {job_id}] Analysis error: {e}")
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )

# ENDPOINT

@app.get("/", tags=["General"])
async def root():
    """API root - returns basic info"""

    return {
        "name": "Financial Analyst API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check health of all system components.

    Returns stats of:
        - API server
        - Ollama server
        - Vector Store (ChromaDB)
        - Tavily
    """
    components = {}
    overall_healthy = True
    
    # Check ollama
    try:
        ollama_healthy = await check_ollama_health()
        components["ollama"] = {
            "status": "healthy" if ollama_healthy else "unhealthy",
            "model": settings.ollama.llm_model,
        }
        if not ollama_healthy:
            overall_healthy = False
    except Exception as e:
        components["ollama"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False

    # Check Vector store
    try:
        store = get_vector_store()
        components["vector_store"] = {
            "status": "healthy",
            "collection": store.collection_name,
        }
    except Exception as e:
        components['vector_store'] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False

    # Check Tavily
    if settings.tavily.api_key:
        components["tavily"] = {"status": "configured"}
    else:
        components["tavily"] = {"status": "not configured"}

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )

@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get system statistics."""
    store = get_vector_store()
    job_stats = job_store.get_stats()

    uptime = (datetime.now(timezone.utc) - SERVER_START_TIME).total_seconds()

    # Count unique tickers in vector store
    unique_companies = 0
    try:
        results = store.collection.get(include=["metadatas"])
        tickers = set()
        for meta in results.get("metadatas") or []:
            if meta and "ticker" in meta:
                tickers.add(meta["ticker"])
        unique_companies = len(tickers)
    except Exception:
        pass

    return StatsResponse(
        total_jobs_completed=job_stats["completed"],
        total_companies_indexed=unique_companies,
        vector_store_documents=store.count,
        uptime_seconds=uptime,
    )

# Analysis Endpoints

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze a company (sync)",
)
async def analyze_company(request: AnalysisRequest):
    """
    Run complete analysis for a company.

    This endpoint blocks until analysis is complete (can take 1-3 minutes).
    For long-running requests, use `/analyze/async` instead.
    """
    logger.info(f"[Sync] Starting analysis for {request.ticker}")

    try:
        result = await run_agent(
            ticker=request.ticker,
            company_name=request.company_name,
        )

        response = state_to_response(result)

        logger.info(f"[Sync] Analysis completed for {request.ticker}")

        return response

    except Exception as e:
        logger.exception(f"[Sync] Analysis failed for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.post(
    "/analyze/async",
    response_model=JobStatusResponse,
    tags=['Analysis'],
    summary="Start analysis job (async)",
)
async def analyze_company_async(request: AnalysisRequest):
    """
    Start analysis job (async).

    Returns immediately with job ID. Poll `/jobs/{job_id}` for status.

    Usage:
        - Post to this endpoint to start job
        - Poll `/jobs/{job_id}` for status is "completed" or "failed"
    """
    job_id = job_store.create_job(request.ticker)
    
    logger.info(f"[Async] Starting analysis job for {request.ticker}")

    # Use asyncio.create_task to run in the same event loop context
    # This avoids issues with httpx async client in BackgroundTasks
    asyncio.create_task(run_analysis_job(job_id, request))

    job = job_store.get_job(job_id)
    started_at = job["started_at"] if job else datetime.now(timezone.utc).isoformat()

    return JobStatusResponse(
        job_id=job_id,
        ticker=request.ticker,
        status=JobStatus.PENDING,
        started_at=started_at,
    )

@app.get(
    "/jobs/{job_id}",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Get job status and results"
)
async def get_job_status(job_id: str):
    """
    Get the status and results of an analysis job.

    Status Values:
        -  `pending`: Job is queued and waiting to run
        -  `running`: Analysis is currently being processed
        -  `completed`: Analysis completed successfully
        -  `failed`: Analysis failed

    When status is "completed", the full analysis response is returned.
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found",
        )

    # if completed, return full response
    if job["status"] == JobStatus.COMPLETED:
        result = job_store.get_result(job_id)
        if result:
            return state_to_response(result, job_id)

    # if failed, return error info
    if job["status"] == JobStatus.FAILED:
        return AnalysisResponse(
            job_id=job_id,
            ticker=job["ticker"],
            company_name="",
            status=JobStatus.FAILED,
            errors=[ErrorDetail(
                step="execution",
                message=job.get("error", "Unknown error"),
                timestamp=job.get("completed_at", ""),
            )],
            started_at=job["started_at"],
            completed_at=job.get("completed_at", ""),
        )

    return AnalysisResponse(
        job_id=job_id,
        ticker=job["ticker"],
        company_name="",
        status=job["status"],
        started_at=job["started_at"],
    )

# Ingestion Endpoints

@app.post(
    "/ingest",
    response_model=IngestionResponse,
    tags=["Ingestion"],
    summary="Ingest SEC filings for a company"
)
async def ingest_filings(request: IngestionRequest):
    """
    Download and index SEC filings for a company.

    This is to be done before analysis can include filing data.

    Process:
        1. Download the latest filing from SEC EDGAR
        2. Extracts key sections
        3. Chunks text for vector search
        4. Stores in ChromaDB
    """
    logger.info(f"[Ingest] Starting ingestion for {request.ticker}")

    try:
        # Delete existing if force refresh
        if request.force_refresh:
            store = get_vector_store()
            deleted = store.delete_by_ticker(request.ticker)
            logger.info(f"[Ingest] Deleted {deleted} existing chunks for {request.ticker}")

        result = await ingest_10k_for_ticker(request.ticker)

        if result.success:
            logger.info(
                f"[Ingest] Success: {result.total_chunks} chunks"
                f" from {len(result.sections_processed)} sections"
            )

            return IngestionResponse(
                ticker=request.ticker,
                filing_type=result.filing_type,
                status="success",
                chunks_created=result.total_chunks,
                sections_processed=result.sections_processed,
                filing_date=result.filing_date,
            )
        else:
            logger.error(f"[Ingest] Failed: {result.error}")

            return IngestionResponse(
                ticker=request.ticker,
                filing_type=request.filing_type,
                status="failed",
                error=result.error,
            )

    except Exception as e:
        logger.exception(f"[Ingest] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )

@app.get(
    "/ingest/{ticker}",
    tags=["Ingestion"],
    summary="Check if filings are ingested for the company",
)
async def check_ingestion(ticker: str):
    """
    Check if SEC filings are ingested for the company.

    Returns:
        The number of chunks ingested if any exist
    """
    store = get_vector_store()

    # Search for any documents with this ticker
    results = store.search(
        query="company business",
        n_results=1,
        filter={"ticker": ticker.upper()},
    )

    if results.has_results:
        # count total chunks for this ticker
        all_results = store.collection.get(
            where={"ticker": ticker.upper()},
            include=[],
        )
        count = len(all_results.get("ids", []))

        return {
            "ticker": ticker.upper(),
            "indexed": True,
            "chunks_count": count,
        }
    else:
        return {
            "ticker": ticker.upper(),
            "indexed": False,
            "chunks_count": 0,
        }

# Main function call
if __name__ == "__main__":
    import uvicorn

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    # Run server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

