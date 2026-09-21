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
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    # or
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, NoReturn, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

# Local Imports
from app.agents.graph import run_agent
from app.agents.state import AgentState
from app.components.retrieval.ingestion import ingest_10k_for_ticker
from app.components.retrieval.vector_store import RetrievalStore, SearchFilters, get_vector_store
from app.config import settings, validate_settings
from app.models import (
    AnalysisRequest,
    AnalysisResponse,
    CitationResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    IngestionRequest,
    IngestionResponse,
    IngestionStatusResponse,
    JobAcceptedResponse,
    JobPollResponse,
    JobStatus,
    NewsArticleResponse,
    SentimentResponse,
    StatsResponse,
    StockDataResponse,
    VerificationClaimResponse,
    VerificationResponse,
)
from app.observability.langsmith import check_langsmith_connection, setup_langsmith_env
from app.observability.metrics import (
    get_metrics,
    get_metrics_content_type,
    track_agent_run,
    track_request,
)
from app.services.llm import check_ollama_health
from app.services.run_store import FileBackedRunStore

RUN_STORE_PATH = Path(".runtime/run_store.json")
run_store = FileBackedRunStore(RUN_STORE_PATH)

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        401: "Missing or invalid API key", 404: "Resource not found",
        405: "Method not allowed",
        409: "Idempotency key conflict", 422: "Request validation failed",
        429: "Submission rate limit exceeded", 500: "Internal server error",
        502: "Upstream ingestion failure", 503: "Service unavailable",
    }.items()
}

bearer_auth = HTTPBearer(auto_error=False)
key_auth = APIKeyHeader(name="X-API-Key", auto_error=False)



class SubmissionRateLimiter:
    """Single-process sliding-window submission limiter with stale-key cleanup."""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, principal: str) -> None:
        now = time.monotonic()
        cutoff = now - settings.api_rate_window_seconds
        # ponytail: single-process limiter; use shared storage before multiple workers.
        self._requests = defaultdict(deque, {
            key: values for key, values in self._requests.items()
            if values and values[-1] > cutoff
        })
        history = self._requests[principal]
        while history and history[0] <= cutoff:
            history.popleft()
        if len(history) >= settings.api_rate_limit:
            retry_after = max(1, int(history[0] + settings.api_rate_window_seconds - now) + 1)
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limited", "message": "Submission rate limit exceeded.", "error_id": _new_error_id()},
                headers={"Retry-After": str(retry_after)},
            )
        history.append(now)



submission_limiter = SubmissionRateLimiter()

def _get_store() -> RetrievalStore:
    """FastAPI dependency. Override in tests via app.dependency_overrides"""
    return get_vector_store()

def _new_error_id() -> str:
    return str(uuid.uuid4())

def _public_error_detail(code: str, message: str, error_id: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "error_id": error_id,
    }

def _public_failure_message(message: str, error_id: str) -> str:
    return f"{message}. error_id={error_id}"

def _raise_internal_error(
    *,
    code: str,
    message: str,
    operation: str,
    exc: Exception,
) -> NoReturn:
    error_id = _new_error_id()
    logger.exception(f"{operation} failed | error_id={error_id}")
    raise HTTPException(
        status_code=500,
        detail=_public_error_detail(code, message, error_id),
    ) from exc


def _error_response(status_code: int, code: str, message: str, *, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": _public_error_detail(code, message, _new_error_id())},
        headers=headers,
    )


async def _authenticate(
    authorization: HTTPAuthorizationCredentials | None = Depends(bearer_auth),
    x_api_key: str | None = Depends(key_auth),
) -> str:
    configured = settings.api_key
    if not configured:
        raise HTTPException(status_code=503, detail={"code": "auth_unconfigured", "message": "API authorization is not configured.", "error_id": _new_error_id()})
    bearer = authorization.credentials if authorization else None
    supplied = x_api_key or bearer
    if supplied is None or not secrets.compare_digest(supplied.encode(), configured.encode()):
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "A valid API key is required.", "error_id": _new_error_id()},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return hashlib.sha256(supplied.encode()).hexdigest()


async def _limit_submission(principal: str = Depends(_authenticate)) -> str:
    submission_limiter.check(principal)
    return principal


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

    # Embeddings always use Ollama, independently of the selected chat provider.
    embeddings_ok = await check_ollama_health(model=settings.ollama.embed_model, log_failure=True)
    if not embeddings_ok:
        logger.warning("Ollama embedding model not available")
    if settings.llm.provider == "ollama":
        await check_ollama_health(log_failure=True)

    try:
        store = get_vector_store()
        logger.info(f"Vector store: {store.count} documents")
    except Exception:
        logger.warning("Vector store unavailable; health endpoint will report degraded")
    logger.info(f"Run store: {RUN_STORE_PATH}")

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
    responses={code: ERROR_RESPONSES[code] for code in (404, 405, 500)},
)


@app.middleware("http")
async def prevent_sensitive_response_caching(request: Request, call_next):
    response = await call_next(request)
    if response.status_code >= 400 or request.url.path.startswith(("/analyze", "/jobs", "/ingest", "/stats")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = [".".join(str(part) for part in error["loc"] if part != "body") for error in exc.errors()]
    return _error_response(422, "validation_error", f"Invalid request fields: {', '.join(fields)}")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and {"code", "message", "error_id"} <= detail.keys():
        content = {"error": detail}
    else:
        content = {"error": _public_error_detail("http_error", str(detail), _new_error_id())}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    error_id = _new_error_id()
    logger.exception(f"Unhandled API error | error_id={error_id}")
    return JSONResponse(status_code=500, content={"error": _public_error_detail("internal_error", "Internal server error.", error_id)}, headers={"Cache-Control": "no-store"})

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "Idempotency-Key"],
    expose_headers=["Location", "Retry-After"],
)


# =============================================================================
# HEALTH & INFO
# =============================================================================


@app.get("/", tags=["Info"], response_model=InfoResponse)
async def root():
    """API information."""
    return {
        "name": "Financial Analyst Agent System",
        "version": "1.1.0",
        "description": "AI-powered financial analysis",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"], responses={503: {"model": HealthResponse, "description": "Dependency unavailable"}})
async def health():
    """Health check endpoint."""
    embedding_ok = await check_ollama_health(model=settings.ollama.embed_model)
    local_chat = settings.llm.provider == "ollama"
    chat_ok = (await check_ollama_health()) if local_chat else bool(
        settings.llm.aws_region and settings.llm.model
    )
    try:
        vector_ok = get_vector_store().count >= 0
    except Exception:
        vector_ok = False
    langsmith_status = await check_langsmith_connection()

    response = HealthResponse(
        status="healthy" if (embedding_ok and chat_ok and vector_ok) else "degraded",
        version="1.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components={
            "chat_model": {"ok": chat_ok, "provider": settings.llm.provider, "check": "model_available" if local_chat else "configuration_only", "inference_verified": False},
            "ollama_embeddings": {"ok": embedding_ok},
            "vector_store": {"ok": vector_ok},
            "langsmith": {"connected": langsmith_status.get("connected", False)},
        },
    )
    return JSONResponse(status_code=200 if response.status == "healthy" else 503, content=response.model_dump())


@app.get("/metrics", tags=["Info"], response_class=Response, responses={**{code: ERROR_RESPONSES[code] for code in (401, 503)}, 200: {"content": {get_metrics_content_type(): {"schema": {"type": "string"}}}}})
async def metrics(_principal: str = Depends(_authenticate)):
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )


@app.get("/stats", tags=["Info"], response_model=StatsResponse, responses={code: ERROR_RESPONSES[code] for code in (401, 503)})
async def stats(_principal: str = Depends(_authenticate), store: RetrievalStore = Depends(_get_store)):
    """Vector store and run-store statistics."""
    return {
        "vector_store": store.get_stats(),
        "run_store": run_store.get_stats(),
    }

# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"], responses={code: ERROR_RESPONSES[code] for code in (401, 422, 429, 503)})
async def analyze(request: AnalysisRequest, _principal: str = Depends(_limit_submission)):
    """
    Run synchronous stock analysis.

    Blocks until analysis is complete.
    """
    ticker = request.ticker.upper()
    logger.info(f"Starting sync analysis for {ticker}")

    with track_request("POST", "/analyze"):
        with track_agent_run(ticker):
            try:
                result = await run_agent(
                    ticker=ticker,
                    company_name=request.company_name,
                    include_filing_analysis=request.include_filing_analysis,
                    include_news_sentiment=request.include_news_sentiment,
                    max_news_articles=request.max_news_articles,
                )
                return _format_response(result)

            except Exception as exc:
                _raise_internal_error(
                    code="analysis_failed",
                    message="Analysis failed.",
                    operation=f"sync analysis for {ticker}",
                    exc=exc,
                )


@app.post("/analyze/async", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Analysis"], responses={**ERROR_RESPONSES, 202: {"headers": {"Location": {"schema": {"type": "string"}, "description": "Polling URL"}}}})
async def analyze_async(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    principal: str = Depends(_limit_submission),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=1, max_length=200),
):
    """
    Start async stock analysis.

    Returns immediately with job_id. Poll /jobs/{job_id} for status.
    """
    ticker = request.ticker.upper()
    job_id = str(uuid.uuid4())

    # Create job
    created = True
    if idempotency_key:
        fingerprint = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        try:
            record, created = run_store.create_idempotent_run(
                job_id, ticker, principal=principal,
                idempotency_key=hashlib.sha256(idempotency_key.encode()).hexdigest(),
                request_fingerprint=fingerprint, company_name=request.company_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "idempotency_conflict", "message": str(exc), "error_id": _new_error_id()}) from exc
    else:
        record = run_store.create_run(job_id=job_id, ticker=ticker, company_name=request.company_name)

    # Start background task
    if created:
        background_tasks.add_task(
        _run_analysis_job,
        record.job_id,
        ticker,
        request.company_name,
        request.include_filing_analysis,
        request.include_news_sentiment,
        request.max_news_articles,
        )
        logger.info(f"Started async job {record.job_id} for {ticker}")

    response.headers["Location"] = f"/jobs/{record.job_id}"

    return JobAcceptedResponse(
        job_id=record.job_id,
        status=JobStatus(record.status),
        ticker=record.ticker,
        started_at=record.started_at,
        error=record.error,
    )


@app.get("/jobs/{job_id}", response_model=JobPollResponse, tags=["Analysis"], responses={code: ERROR_RESPONSES[code] for code in (401, 422, 503)})
async def get_job_status(job_id: uuid.UUID, _principal: str = Depends(_authenticate)):
    """Get async job status and return completed result when available."""
    record = run_store.get_run(str(job_id))
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found.", "error_id": _new_error_id()})

    result = AnalysisResponse.model_validate(record.result) if record.result else None

    return JobPollResponse(
        job_id=record.job_id,
        status=JobStatus(record.status),
        ticker=record.ticker,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
        result=result,
    )


async def _run_analysis_job(
    job_id: str,
    ticker: str,
    company_name: Optional[str],
    include_filing_analysis: bool = True,
    include_news_sentiment: bool = True,
    max_news_articles: int = 10,
) -> None:
    """Background task for async analysis."""
    run_store.mark_running(job_id)

    try:
        with track_agent_run(ticker):
            result = await run_agent(
                ticker,
                company_name,
                include_filing_analysis=include_filing_analysis,
                include_news_sentiment=include_news_sentiment,
                max_news_articles=max_news_articles,
            )
            formatted = _format_response(result).model_dump()
            formatted["job_id"] = job_id
            run_store.mark_finished(job_id, status=formatted["status"], result=formatted)
    except Exception:
        error_id = _new_error_id()
        logger.exception(
            f"Async analysis job failed | job_id={job_id} | ticker={ticker} | error_id={error_id}"
        )
        run_store.mark_failed(
            job_id,
            error=_public_failure_message(message="Analysis job failed", error_id=error_id),
        )

def _format_response(state: AgentState) -> AnalysisResponse:
    """Format agent state as API response."""
    stock = state.get("stock_data", {})
    sentiment = state.get("sentiment_result", {})
    verification = state.get("verification_result", {})

    errors = [
        ErrorDetail(
            step=error.get("step", ""),
            message="A workflow step failed.",
            timestamp=error.get("timestamp", ""),
            recoverable=error.get("recoverable", True),
        )
        for error in state.get("errors", [])
    ]

    missing: list[str] = []
    if not stock:
        missing.append("stock")
    if state.get("include_filing_analysis", True) and not state.get("filing_chunks"):
        missing.append("filings")
    if state.get("include_news_sentiment", True) and not state.get("news_articles"):
        missing.append("news")
    if state.get("include_news_sentiment", True) and not sentiment:
        missing.append("sentiment")
    if any(not error.recoverable for error in errors) or not state.get("investment_memo"):
        response_status = JobStatus.FAILED
    elif missing:
        response_status = JobStatus.EVIDENCE_MISSING
    elif verification.get("passed") is not True:
        response_status = JobStatus.DEGRADED
    else:
        response_status = JobStatus.COMPLETED

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
        status=response_status,
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
                title=article.get("title", ""),
                url=article.get("url", ""),
                source=article.get("source", "Unknown"),
                snippet=article.get("snippet", ""),
                published_date=article.get("published_date"),
                relevance_score=article.get("relevance_score", 0.0),
            )
            for article in state.get("news_articles", [])
        ],
        citations=[
            CitationResponse(
                index=citation.get("index", 0),
                source_type=citation.get("source_type", ""),
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
        missing=missing,
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


@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"], responses={code: ERROR_RESPONSES[code] for code in (401, 422, 429, 502, 503)})
async def ingest_filing(request: IngestionRequest, _principal: str = Depends(_limit_submission)):
    """Ingest SEC filing for a ticker."""
    ticker = request.ticker.upper()
    logger.info(f"Ingesting 10-K for {ticker}")
    filing_type = "10-K"

    with track_request("POST", "/ingest"):
        try:
            result = await ingest_10k_for_ticker(ticker, replace_existing=request.force_refresh)
            sections_processed = _normalize_sections_processed(
                getattr(result, "sections_processed", None)
            )

            if not result.success:
                raise HTTPException(status_code=502, detail={"code": "ingestion_failed", "message": "Filing ingestion failed.", "error_id": _new_error_id()})
            return IngestionResponse(
                ticker=ticker,
                filing_type=filing_type,
                status="success" if result.success else "failed",
                chunks_created=getattr(result, "total_chunks", 0),
                sections_processed=sections_processed,
                filing_date=getattr(result, "filing_date", None),
                error=None,
            )

        except HTTPException:
            raise
        except Exception:
            error_id = _new_error_id()
            logger.exception(
                f"Ingestion failed | ticker={ticker} | error_id={error_id}",
            )
            raise HTTPException(status_code=502, detail={"code": "ingestion_failed", "message": "Filing ingestion failed.", "error_id": error_id})


@app.get("/ingest/{ticker}", tags=["Ingestion"], response_model=IngestionStatusResponse, responses={code: ERROR_RESPONSES[code] for code in (401, 422, 503)})
async def check_ingestion(
    ticker: Annotated[str, ApiPath(pattern=r"^[A-Za-z]{1,5}(?:\.[A-Za-z])?$")],
    _principal: str = Depends(_authenticate),
    store: RetrievalStore = Depends(_get_store),
):
    """Check if a ticker has been ingested."""
    ticker = ticker.upper()
    document_count = store.count_documents(SearchFilters(ticker=ticker))

    return {
        "ticker": ticker,
        "indexed": document_count > 0,
        "document_count": document_count,
    }


# =============================================================================
# CLI
# =============================================================================


def run_api() -> None:
    """Console entrypoint for the `financial-analyst-agent-system` script."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_api()
