// Mirrors app/models.py (Pydantic). Keep in sync by hand.

export type JobStatus = "pending" | "running" | "completed" | "degraded" | "evidence_missing" | "failed";

export const isTerminalJobStatus = (status: JobStatus): boolean => status !== "pending" && status !== "running";
export type SentimentLabel = "positive" | "negative" | "neutral" | "mixed";

export interface AnalysisRequest {
  ticker: string;
  company_name?: string | null;
  include_filing_analysis?: boolean;
  include_news_sentiment?: boolean;
  max_news_articles?: number;
}

export interface NewsArticleResponse {
  title: string;
  url: string;
  source: string;
  snippet: string;
  published_date?: string | null;
  relevance_score: number;
  sentiment?: string | null;
  sentiment_confidence?: number | null;
}

export interface StockDataResponse {
  ticker: string;
  company_name: string;
  current_price?: number | null;
  price_change_percent?: number | null;
  market_cap?: number | null;
  market_cap_formatted?: string | null;
  pe_ratio?: number | null;
  fifty_two_week_high?: number | null;
  fifty_two_week_low?: number | null;
  target_price?: number | null;
  sector?: string | null;
  industry?: string | null;
}

export interface SentimentResponse {
  overall_sentiment: string;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  average_positive_score: number;
  average_negative_score: number;
}

export interface CitationResponse {
  index: number;
  source_type: string;
  title: string;
  url?: string | null;
  date?: string | null;
}

export interface VerificationClaimResponse {
  sentence: string;
  citations: number[];
  supported: boolean;
  overlap_score: number;
  missing_citation: boolean;
  reason?: string | null;
}

export interface VerificationResponse {
  passed: boolean;
  total_claims: number;
  cited_claims: number;
  grounded_claims: number;
  citation_coverage_rate: number;
  grounded_claim_rate: number;
  claims: VerificationClaimResponse[];
  orphan_citations: number[];
}

export interface UsageResponse {
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
}

/** One workflow node's execution; `ended_at`/`duration_ms` are absent while it runs. */
export interface NodeProgress {
  node: string;
  status: "running" | "completed" | "degraded";
  started_at: string;
  ended_at?: string | null;
  duration_ms?: number | null;
}

export interface ErrorDetail {
  step: string;
  message: string;
  timestamp: string;
  recoverable: boolean;
}

export interface AnalysisResponse {
  job_id?: string | null;
  request_id?: string | null;
  trace_id?: string | null;
  ticker: string;
  company_name: string;
  status: JobStatus;
  executive_summary?: string | null;
  investment_memo?: string | null;
  filing_chunk_count?: number | null;
  usage?: UsageResponse | null;
  stock_data?: StockDataResponse | null;
  sentiment?: SentimentResponse | null;
  news_articles: NewsArticleResponse[];
  citations: CitationResponse[];
  verification?: VerificationResponse | null;
  errors: ErrorDetail[];
  missing: string[];
  started_at?: string | null;
  completed_at?: string | null;
  execution_time_ms?: number | null;
}

export interface JobAcceptedResponse {
  job_id: string;
  request_id?: string | null;
  trace_id?: string | null;
  ticker: string;
  status: JobStatus;
  started_at: string;
  error?: string | null;
}

export interface JobPollResponse {
  job_id: string;
  request_id?: string | null;
  trace_id?: string | null;
  ticker: string;
  status: JobStatus;
  started_at: string;
  completed_at?: string | null;
  error?: string | null;
  progress?: NodeProgress[];
  result?: AnalysisResponse | null;
}

/** One stored run from GET /jobs (no memo or evidence). */
export interface RunSummary {
  job_id: string;
  ticker: string;
  status: JobStatus;
  started_at: string;
  completed_at?: string | null;
  grounded_claim_rate?: number | null;
  citation_coverage_rate?: number | null;
  execution_time_ms?: number | null;
}

/** Body of every non-2xx response from our own /api/* handlers. */
export interface ApiError {
  error: string;
}
