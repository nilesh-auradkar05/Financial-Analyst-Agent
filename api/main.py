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
GET /jobs/{job_id}       - Get job status and completed result
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

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Local Imports
from agents.graph import run_agent
from agents.state import AgentState
from api.run_store import FileBackedRunStore
from api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    CitationResponse,
    ErrorDetail,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    JobAcceptedResponse,
    JobPollResponse,
    JobStatus,
    NewsArticleResponse,
    SentimentResponse,
    StockDataResponse,
    VerificationClaimResponse,
    VerificationResponse,
)
from configs.config import settings, validate_settings
from models.llm import check_ollama_health
from observability.langsmith import check_langsmith_connection, setup_langsmith_env
from observability.metrics import (
    get_metrics,
    get_metrics_content_type,
    track_agent_run,
    track_request,
)
from rag.ingestion import ingest_10k_for_ticker
from rag.vector_store import ChromaDBVectorStore, get_vector_store

RUN_STORE_PATH = Path(".runtime/run_store.json")
run_store = FileBackedRunStore(RUN_STORE_PATH)

def _get_store() -> ChromaDBVectorStore:
    """FastAPI dependency: Override in tests via app.dependency_overrides"""
    return get_vector_store()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Financial Analyst Agent System API...")

    # Validate settings
    warnings = validate_settings()
    for w in warnings:
        logger.warning(w)

    # Setup LangSmith
    setup_langsmith_env()

    # Check Ollama
    ollama_ok = await check_ollama_health()
    if ollama_ok:
        logger.info(f"Ollama connected ({settings.ollama.llm_model})")
    else:
        logger.warning("Ollama not available")

    # Initialize vector store
    store = get_vector_store()
    logger.info(f"Run store: {RUN_STORE_PATH}")
    logger.info(f"Vector store: {store.count} documents")

    logger.info("Financial Analyst Agent System API ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")


# =============================================================================
# APP
# =============================================================================


app = FastAPI(
    title="Financial Analyst Agent System",
    description="AI-powered financial analysis agent",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# =============================================================================
# HEALTH & INFO
# =============================================================================


@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Financial Analyst Agent System",
        "version": "1.1.0",
        "description": "AI-powered financial analysis",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health(store: ChromaDBVectorStore = Depends(_get_store)):
    """Health check endpoint."""
    ollama_ok = await check_ollama_health()
    vector_ok = store.count >= 0
    langsmith_status = await check_langsmith_connection()

    return HealthResponse(
        status="healthy" if (ollama_ok and vector_ok) else "degraded",
        version="1.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components={
            "ollama": {"ok": ollama_ok},
            "vector_store": {"ok": vector_ok},
            "langsmith": {"connected": langsmith_status.get("connected", False)},
        }
    )


@app.get("/metrics", tags=["Info"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )


@app.get("/stats", tags=["Info"])
async def stats(store: ChromaDBVectorStore = Depends(_get_store)):
    """Vector store statistics."""
    return {
        "vector_store": store.get_stats(),
        "run_store": run_store.get_stats(),
    }

# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(request: AnalysisRequest):
    """
    Run synchronous stock analysis.

    Blocks until analysis is complete (may take 1-3 minutes).
    """
    ticker = request.ticker.upper()
    logger.info(f"Starting sync analysis for {ticker}")

    with track_request("POST", "/analyze"):
        with track_agent_run(ticker):
            try:
                result = await run_agent(ticker, request.company_name)
                return _format_response(result)

            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/async", response_model=JobAcceptedResponse, tags=["Analysis"])
async def analyze_async(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start async stock analysis.

    Returns immediately with job_id. Poll /jobs/{job_id} for status.
    """
    ticker = request.ticker.upper()
    job_id = str(uuid.uuid4())

    # Create job
    record = run_store.create_run(
        job_id=job_id,
        ticker=ticker,
        company_name=request.company_name,
    )

    # Start background task
    background_tasks.add_task(_run_analysis_job, job_id, ticker, request.company_name)
    logger.info(f"Started async job {job_id} for {ticker}")

    return JobAcceptedResponse(
        job_id=record.job_id,
        status=record.status,
        ticker=record.ticker,
        started_at=record.started_at,
        error=record.error,
    )


@app.get("/jobs/{job_id}", response_model=JobPollResponse, tags=["Analysis"])
async def get_job_status(job_id: str):
    """Get async job status and return completed result when available."""
    record = run_store.get_run(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result = AnalysisResponse.model_validate(record.result) if record.result else None

    return JobPollResponse(
        job_id=record.job_id,
        status=record.status,
        ticker=record.ticker,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
        result=result,
    )


async def _run_analysis_job(job_id: str, ticker: str, company_name: Optional[str]):
    """Background task for async analysis."""
    run_store.mark_running(job_id)

    try:
        with track_agent_run(ticker):
            result = await run_agent(ticker, company_name)
            formatted = _format_response(result).model_dump()
            run_store.mark_completed(job_id, result=formatted)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        run_store.mark_failed(job_id, error=str(e))

def _format_response(state: AgentState) -> AnalysisResponse:
    """Format agent state as API response."""
    stock = state.get("stock_data", {})
    sentiment = state.get("sentiment_result", {})
    verification = state.get("verification_result", {})

    errors = [
        ErrorDetail(
            step=error.get("step", ""),
            message=error.get("message", ""),
            timestamp=error.get("timestamp", ""),
            recoverable=error.get("recoverable", True),
        )
        for error in state.get("errors", [])
    ]

    if state.get("investment_memo"):
        status = JobStatus.COMPLETED
    elif errors:
        status = (JobStatus.FAILED if not any(error.recoverable for error in errors) else JobStatus.COMPLETED)
    else:
        status = JobStatus.FAILED

    market_cap_formatted = None
    market_cap = stock.get("market_cap")
    if market_cap:
        if market_cap >= 1_000_000_000_000:
            market_cap_formatted = f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            market_cap_formatted = f"${market_cap / 1_000_000_000:.2f}B"
        else:
            market_cap_formatted = f"${market_cap / 1_000_000:.2f}M"

    return AnalysisResponse(
        ticker=state.get("ticker", ""),
        company_name=state.get("company_name", ""),
        status=status,
        executive_summary=state.get("executive_summary"),
        investment_memo=state.get("investment_memo"),
        stock_data=StockDataResponse(
            ticker=stock.get("ticker", ""),
            company_name=stock.get("company_name", ""),
            current_price=stock.get("current_price"),
            price_change_percent=stock.get("price_change_percent"),
            market_cap=stock.get("market_cap"),
            market_cap_formatted=market_cap_formatted,
            pe_ratio=stock.get("pe_ratio"),
            fifty_two_week_high=stock.get("fifty_two_week_high"),
            fifty_two_week_low=stock.get("fifty_two_week_low"),
            target_price=stock.get("target_price"),
            sector=stock.get("sector"),
            industry=stock.get("industry"),
        ) if stock else None,
        sentiment=SentimentResponse(
            overall_sentiment=sentiment.get("overall_sentiment", "neutral"),
            positive_count=sentiment.get("positive_count", 0),
            negative_count=sentiment.get("negative_count", 0),
            neutral_count=sentiment.get("neutral_count", 0),
            average_positive_score=sentiment.get("average_positive_score", 0.0),
            average_negative_score=sentiment.get("average_negative_score", 0.0),
        ) if sentiment else None,
        news_articles=[
            NewsArticleResponse(
                title=a.get("title", ""),
                url=a.get("url", ""),
                source=a.get("source", "Unknown"),
                snippet=a.get("snippet", ""),
                published_date=a.get("published_date"),
                relevance_score=a.get("relevance_score", 0.0),
            )
            for a in state.get("news_articles", [])
        ],
        citations=[
            CitationResponse(
                index=citation.get("index", 0),
                source_type=citation.get("source_type") or citation.get("type", ""),
                title=citation.get("title", ""),
                url=citation.get("url"),
                date=citation.get("date"),
            )
            for citation in state.get("citations", [])
        ],
        verification=VerificationResponse(
            passed=verification.get("passed", False),
            total_claims=verification.get("total_claims", 0),
            cited_claims=verification.get("cited_claims", 0),
            grounded_claims=verification.get("grounded_claims", 0),
            citation_coverage_rate=verification.get("citation_coverage_rate", 0.0),
            grounded_claim_rate=verification.get("grounded_claim_rate", 0.0),
            orphan_citations=verification.get("orphan_citations", []),
            claims=[
                VerificationClaimResponse(**claim)
                for claim in verification.get("claims", [])
            ],
        ) if verification else None,
        errors=errors,
        started_at=state.get("started_at"),
        completed_at=state.get("completed_at"),
        execution_time_ms=state.get("execution_time_ms"),
    )

def _normalize_sections_processed(value: object) -> list[str]:
    """Normalize ingestion section metadata into the API contract shape.

    Expected public shape: list[str]

    We accept a few legacy/internal shapes defensively so that
    integration tests and older ingestion stubs do not crash the API.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple | set):
        return [str(item) for item in value]

    if isinstance(value, int):
        # Legacy/mocked shape: only a count is available.
        # Keep the API stable without inventing fake section names.
        return []

    return [str(value)]


# =============================================================================
# INGESTION ENDPOINTS
# =============================================================================


@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_filing(request: IngestionRequest):
    """Ingest SEC filing for a ticker."""
    ticker = request.ticker.upper()
    logger.info(f"Ingesting 10-K for {ticker}")
    filing_type = "10-K"

    with track_request("POST", "/ingest"):
        try:
            result = await ingest_10k_for_ticker(ticker)
            sections_processed = _normalize_sections_processed(
                getattr(result, "sections_processed", None)
            )

            return IngestionResponse(
                ticker=ticker,
                filing_type=filing_type,
                status="success" if result.success else "failed",
                chunks_created=getattr(result, "total_chunks", 0),
                sections_processed=sections_processed,
                filing_date=getattr(result, "filing_date", None),
                error=getattr(result, "error", None),
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestionResponse(
                ticker=ticker,
                filing_type=filing_type,
                status="failed",
                chunks_created=0,
                sections_processed=[],
                error=str(e),
            )


@app.get("/ingest/{ticker}", tags=["Ingestion"])
async def check_ingestion(ticker: str, store: ChromaDBVectorStore = Depends(_get_store)):
    """Check if a ticker has been ingested."""
    ticker = ticker.upper()
    result = store.search_by_ticker("business", ticker, n_results=1)

    return {
        "ticker": ticker,
        "indexed": result.has_results,
        "document_count": len(result.chunks) if result.has_results else 0,
    }


# =============================================================================
# CLI
# =============================================================================


def run_api() -> None:
    """Console entrypoint for the `alpha-analyst` script (see pyproject [project.scripts])."""
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_api()
