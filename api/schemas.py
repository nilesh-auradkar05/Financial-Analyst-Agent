"""
API Schemas

This module defines the Pydantic models for the API request/response validation.
"""

from email.policy import default
from optparse import Option
from re import S
from turtle import st
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum

# ENUMS

class JobStatus(str, Enum):
    """Status of an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

# REQUEST MODELS

class AnalysisRequest(BaseModel):
    """Request to analyze a company.
    
    Attributes:
        - ticker: Stock ticker symbol (eg: AAPL)
        - company_name: Company name (Optional - Auto-fetched if not provided)
        - include_filing_analysis: Whether to include SEC filing analysis.
        - include_news_analysis: Whether to include news sentiment.
        - max_news_articles: Maximum news articles to analyze
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (eg: 'AAPL', 'TSLA')",
        examples=["AAPL", "TSLA"],
    )

    company_name: Optional[str] = Field(
        default=None,
        description="Company name (Optional, auto-fetched if not provided)",
    )

    include_filing_analysis: bool = Field(
        default=True,
        description="Include SEC filing analysis in the memo.",
    )

    include_news_sentiment: bool = Field(
        default=True,
        description="Include news sentiment analysis.",
    )

    max_news_articles: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of news articles to analyze.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "include_filing_analysis": True,
                "include_news_sentiment": True,
                "max_news_articles": 5,
            }
        }

class IngestionRequest(BaseModel):
    """Request to ingest SEC filings for a company.
    
    Must be done before analysis can include filing data.
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol",
    )

    filing_type: str = Field(
        default="10-K",
        description="Types of filing to ingest",
    )

    force_refresh: bool = Field(
        default=False,
        description="Re-download and re-index even if already ingested."
    )

# RESPONSE MODELS

class NewsArticleResponse(BaseModel):
    """News Article in API response."""

    title: str
    url: str
    source: str
    snippet: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None

class StockDataResponse(BaseModel):
    """Stock market data in API response."""

    ticker: str
    company_name: str
    current_price: Optional[float] = None
    price_change_percent: Optional[float] = None
    market_cap: Optional[int] = None
    market_cap_formatted: Optional[str] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    target_price: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

class SentimentResponse(BaseModel):
    """Sentiment analysis result in API response."""

    overall_sentiment: str
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_positive_score: float = 0.0
    average_negative_score: float = 0.0

class CitationResponse(BaseModel):
    """Citation in API response."""

    index: int
    source_type: str
    title: str
    url: Optional[str] = None
    date: Optional[str] = None

class ErrorDetail(BaseModel):
    """Error information."""

    step: str
    message: str
    timestamp: str
    recoverable: bool = True

# Response MODELS

class AnalysisResponse(BaseModel):
    """Complete analysis response.
    
    Returned when analysis is complete for sync jobs or when fetching completed job results for async jobs.
    """

    # Identifiers
    job_id: Optional[str] = None
    ticker: str
    company_name: str

    # Status
    status: JobStatus

    # Results
    executive_summary: Optional[str] = None
    investment_memo: Optional[str] = None
    
    # Supporting data
    stock_data: Optional[StockDataResponse] = None
    sentiment: Optional[SentimentResponse] = None
    news_articles: list[NewsArticleResponse] = []
    citations: list[CitationResponse] = []

    # Errors (for partial failures)
    errors: list[ErrorDetail] = []

    # Metadata
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123",
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "status": "completed",
                "executive_summary": "Apple Inc shows strong fundamentals...",
                "investment_memo": "# Investment Memo: Apple Inc...",
                "execution_time_ms": 45000,
            }
        }

class JobStatusResponse(BaseModel):
    """Status of an analysis job.

    Used for polling async job status.
    """

    job_id: str
    ticker: str
    status: JobStatus
    current_step: Optional[str] = None
    progress_percent: Optional[float] = None
    started_at: str
    estimated_completion_time: Optional[str] = None
    error: Optional[str] = None

class IngestionResponse(BaseModel):
    """Response for filing ingestion request."""

    ticker: str
    filing_type: str
    status: str
    chunks_created: int = 0
    sections_processed: list[str] = []
    filing_date: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response for API."""

    status: str
    version: str
    timestamp: str
    components: dict[str, dict]

class StatsResponse(BaseModel):
    """System statistics response."""

    total_jobs_completed: int
    total_companies_indexed: int
    vector_store_documents: int
    uptime_seconds: float

# ERROR RESPONSES

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Analysis failed",
                "detail": "Could not connect to Ollama server",
                "status_code": 500,
                "timestamp": "2025-01-01T10:32:00Z",
            }
        }

class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    error: str = "Validation error"
    detail: list[dict]
    status_code: int = 422

# MODULE EXPORTS

__all__ = [
    # ENUMS
    "JobStatus",
    "SentimentLabel",

    # Requests
    "AnalysisRequest",
    "IngestionRequest",

    # Response components
    "NewsArticleResponse",
    "StockDataResponse",
    "SentimentResponse",
    "CitationResponse",
    "ErrorDetail",

    # Main responses
    "AnalysisResponse",
    "JobStatusResponse",
    "IngestionResponse",
    "HealthResponse",
    "StatsResponse",

    # Errors
    "ErrorResponse",
    "ValidationErrorResponse",
]
