from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as api_main
from app.components.retrieval.vector_store import SearchFilters
from app.services.run_store import FileBackedRunStore


class StubSearchResult:
    def __init__(self, has_results: bool = True, chunks: list[dict[str, Any]] | None = None):
        self.has_results = has_results
        self.chunks = chunks or [{"chunk_id": "chunk-1"}]


class StubVectorStore:
    def __init__(self) -> None:
        self.count = 7
        self.tickers = {"AAPL", "MSFT"}

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "stub",
            "document_count": self.count,
            "tickers": sorted(self.tickers),
        }

    def count_documents(self, filters: SearchFilters | None = None) -> int:
        if filters is None:
            return self.count

        if filters.ticker and filters.ticker.upper() not in self.tickers:
            return 0

        return self.count

    def search_by_ticker(
        self,
        query: str,
        ticker: str,
        n_results: int = 1,
    ) -> StubSearchResult:
        return StubSearchResult(
            has_results=True,
            chunks=[
                {
                    "chunk_id": f"{ticker}-chunk-1",
                    "query": query,
                    "ticker": ticker,
                }
            ],
        )


class StubIngestResult:
    success = True
    total_chunks = 8
    sections_processed = ["business", "risk_factors", "md&a"]
    filing_date = "2025-09-28"
    error: str | None = None


async def fake_check_ollama_health(*, log_failure: bool = False, model: str | None = None) -> bool:
    return True


async def fake_check_langsmith_connection() -> dict[str, bool]:
    return {"connected": True}


async def fake_ingest_10k_for_ticker(ticker: str, replace_existing: bool = False) -> StubIngestResult:
    return StubIngestResult()


async def fake_run_agent(
    ticker: str,
    company_name: str | None,
    include_filing_analysis: bool = True,
    include_news_sentiment: bool = True,
    max_news_articles: int = 10,
):
    company = company_name or {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
    }.get(ticker, f"{ticker} Corp.")

    if not include_filing_analysis or not include_news_sentiment:
        return {
            "ticker": ticker,
            "company_name": company,
            "executive_summary": "",
            "investment_memo": "",
            "stock_data": {},
            "sentiment_result": {},
            "news_articles": [],
            "citations": [],
            "errors": [],
        }

    return {
        "ticker": ticker,
        "company_name": company,
        "executive_summary": f"{company} remains financially solid.",
        "investment_memo": f"# Investment Memo\n\nBull case for {ticker}.",
        "stock_data": {
            "ticker": ticker,
            "company_name": company,
            "current_price": 189.12,
            "price_change_percent": 1.42,
            "market_cap": 3_000_000_000_000,
            "pe_ratio": 31.8,
            "fifty_two_week_high": 199.62,
            "fifty_two_week_low": 164.08,
            "target_price": 205.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        "sentiment_result": {
            "overall_sentiment": "positive",
            "positive_count": 4,
            "negative_count": 1,
            "neutral_count": 0,
            "average_positive_score": 0.84,
            "average_negative_score": 0.22,
        },
        "news_articles": [
            {
                "title": f"{company} launches new product line",
                "url": f"https://example.com/{ticker.lower()}-launch",
                "source": "Example News",
                "snippet": "Launch expands addressable market.",
                "published_date": "2026-03-15",
                "relevance_score": 0.93,
            }
        ],
        "citations": [
            {
                "index": 1,
                "source_type": "news",
                "title": f"{company} launches new product line",
                "url": f"https://example.com/{ticker.lower()}-launch",
                "date": "2026-03-15",
            }
        ],
        "filing_chunks": [{"text": "filing evidence"}],
        "verification_result": {
            "passed": True,
            "total_claims": 2,
            "cited_claims": 2,
            "grounded_claims": 2,
            "citation_coverage_rate": 1.0,
            "grounded_claim_rate": 1.0,
            "claims": [],
            "orphan_citations": [],
        },
        "errors": [],
        "started_at": "2026-03-15T12:00:00+00:00",
        "completed_at": "2026-03-15T12:00:02+00:00",
        "execution_time_ms": 2000,
    }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_store = FileBackedRunStore(tmp_path / "run_store.json")
    monkeypatch.setattr(api_main, "run_store", temp_store)
    monkeypatch.setattr(api_main, "get_vector_store", lambda: StubVectorStore())
    monkeypatch.setattr(api_main, "check_ollama_health", fake_check_ollama_health)
    monkeypatch.setattr(api_main, "check_langsmith_connection", fake_check_langsmith_connection)
    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", fake_ingest_10k_for_ticker)
    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    monkeypatch.setattr(api_main.settings, "api_key", "test-secret")
    monkeypatch.setattr(api_main, "submission_limiter", api_main.SubmissionRateLimiter())

    with TestClient(api_main.app) as test_client:
        test_client.headers["Authorization"] = "Bearer test-secret"
        yield test_client

def test_health_endpoint_returns_component_statuses(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["components"]["ollama_embeddings"]["ok"] is True
    assert payload["components"]["vector_store"]["ok"] is True
    assert payload["components"]["langsmith"]["connected"] is True


def test_ingest_and_ingest_status_endpoints(client: TestClient):
    ingest_response = client.post("/ingest", json={"ticker": "AAPL"})

    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert ingest_payload["ticker"] == "AAPL"
    assert ingest_payload["filing_type"] == "10-K"
    assert ingest_payload["status"] == "success"
    assert ingest_payload["chunks_created"] == 8
    assert ingest_payload["sections_processed"] == ["business", "risk_factors", "md&a"]

    status_response = client.get("/ingest/AAPL")

    assert status_response.status_code == 200
    assert status_response.json() == {
        "ticker": "AAPL",
        "indexed": True,
        "document_count": 7,
    }


def test_sync_analysis_endpoint_returns_full_analysis_payload(client: TestClient):
    response = client.post(
        "/analyze",
        json={"ticker": "AAPL", "company_name": "Apple Inc."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["company_name"] == "Apple Inc."
    assert payload["status"] == "completed"
    assert payload["executive_summary"] == "Apple Inc. remains financially solid."
    assert payload["investment_memo"].startswith("# Investment Memo")
    assert payload["stock_data"]["market_cap_formatted"] == "$3.00T"
    assert payload["sentiment"]["overall_sentiment"] == "positive"
    assert payload["verification"] is not None
    assert payload["verification"]["passed"] is True
    assert payload["verification"]["citation_coverage_rate"] == 1.0
    assert payload["verification"]["grounded_claim_rate"] == 1.0
    assert payload["verification"]["orphan_citations"] == []
    assert len(payload["news_articles"]) == 1
    assert len(payload["citations"]) == 1
    assert payload["errors"] == []


def test_async_analysis_poll_returns_completed_result_payload(client: TestClient):
    create_response = client.post(
        "/analyze/async",
        json={"ticker": "MSFT", "company_name": "Microsoft Corp."},
    )

    assert create_response.status_code == 202
    assert create_response.headers["location"].startswith("/jobs/")
    create_payload = create_response.json()
    assert create_payload["ticker"] == "MSFT"
    assert create_payload["status"] in {"pending", "running", "completed"}

    job_id = create_payload["job_id"]
    status_response = client.get(f"/jobs/{job_id}")

    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["job_id"] == job_id
    assert status_payload["ticker"] == "MSFT"
    assert status_payload["status"] == "completed"
    assert status_payload["result"] is not None
    assert status_payload["result"]["ticker"] == "MSFT"
    assert status_payload["result"]["investment_memo"].startswith("# Investment Memo")
    assert status_payload["result"]["verification"] is not None
    assert status_payload["result"]["verification"]["passed"] is True

    record = api_main.run_store.get_run(job_id)
    assert record is not None
    assert record.status == "completed"
    assert record.result is not None


def test_analysis_request_accepts_optional_controls(client: TestClient):
    response = client.post(
        "/analyze",
        json={
            "ticker": "AAPL",
            "include_filing_analysis": True,
            "include_news_sentiment": True,
            "max_news_articles": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["status"] == "completed"


def test_analysis_request_controls_disable_filing_analysis(client: TestClient):
    response = client.post(
        "/analyze",
        json={"ticker": "AAPL", "include_filing_analysis": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert all(c["source_type"] != "sec_filing" for c in payload["citations"])


def test_analysis_request_controls_disable_news_sentiment(client: TestClient):
    response = client.post(
        "/analyze",
        json={"ticker": "AAPL", "include_news_sentiment": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment"] is None


def test_analysis_request_controls_limit_news_articles(client: TestClient):
    response = client.post(
        "/analyze",
        json={"ticker": "AAPL", "max_news_articles": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["news_articles"]) <= 1


def test_ingestion_request_accepts_optional_controls(client: TestClient):
    response = client.post(
        "/ingest",
        json={"ticker": "AAPL", "filing_type": "10-K", "force_refresh": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["filing_type"] == "10-K"
    assert payload["status"] == "success"


def test_stats_endpoint_reports_vector_and_run_store_counts(client: TestClient):
    response = client.post(
        "/analyze/async",
        json={"ticker": "AAPL", "company_name": "Apple Inc."},
    )
    assert response.status_code == 202

    stats_response = client.get("/stats")

    assert stats_response.status_code == 200
    payload = stats_response.json()
    assert payload["vector_store"] == {
        "backend": "stub",
        "document_count": 7,
        "tickers": ["AAPL", "MSFT"],
    }
    assert payload["run_store"]["total_runs"] == 1
    assert payload["run_store"]["completed"] == 1
    assert payload["run_store"]["failed"] == 0


def test_sync_analysis_failure_returns_stable_public_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    async def failing_run_agent(*args, **kwargs):
        raise RuntimeError("secret provider token exploded")

    monkeypatch.setattr(api_main, "run_agent", failing_run_agent)

    response = client.post("/analyze", json={"ticker": "AAPL"})

    assert response.status_code == 500
    detail = response.json()["error"]
    assert detail["code"] == "analysis_failed"
    assert detail["message"] == "Analysis failed."
    assert detail["error_id"]
    assert "secret provider token exploded" not in str(detail)


def test_protected_routes_require_auth_but_root_and_health_are_public(client: TestClient):
    client.headers.pop("Authorization")
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    for method, path in [("get", "/stats"), ("get", "/metrics"), ("get", "/ingest/AAPL")]:
        response = getattr(client, method)(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["cache-control"] == "no-store"


def test_input_contract_normalizes_ticker_and_rejects_unsafe_values(client: TestClient):
    assert client.post("/analyze", json={"ticker": "brk.b"}).json()["ticker"] == "BRK.B"
    assert client.post("/analyze", json={"ticker": "AAPL; DROP"}).status_code == 422
    assert client.post("/analyze", json={"ticker": "AAPL", "company_name": "bad\nname"}).status_code == 422
    assert client.post("/ingest", json={"ticker": "AAPL", "filing_type": "10-Q"}).status_code == 422


def test_async_idempotency_reuses_job_and_conflicts_on_changed_body(client: TestClient):
    headers = {"Idempotency-Key": "request-1"}
    first = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=headers)
    same = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=headers)
    changed = client.post("/analyze/async", json={"ticker": "MSFT"}, headers=headers)
    assert first.status_code == same.status_code == 202
    assert first.json()["job_id"] == same.json()["job_id"]
    assert changed.status_code == 409


def test_method_error_preserves_allow_header(client: TestClient):
    response = client.put("/analyze", json={"ticker": "AAPL"})
    assert response.status_code == 405
    assert "POST" in response.headers["allow"]


def test_submission_rate_limit_returns_retry_after(client: TestClient):
    for _ in range(10):
        assert client.post("/analyze", json={"ticker": "AAPL"}).status_code == 200
    response = client.post("/analyze", json={"ticker": "AAPL"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1


def test_openapi_documents_async_and_error_contracts(client: TestClient):
    operation = client.get("/openapi.json").json()["paths"]["/analyze/async"]["post"]
    assert "202" in operation["responses"]
    assert {"401", "409", "422", "429", "500"} <= operation["responses"].keys()


# Trace: docs/test-plan.md §1 REST hardening and §11 evidence completeness.
@pytest.mark.parametrize("patch, expected, missing", [
    ({"verification_result": {"passed": False}}, "degraded", []),
    ({"verification_result": {}}, "degraded", []),
    ({"filing_chunks": []}, "evidence_missing", ["filings"]),
    ({"stock_data": {}}, "evidence_missing", ["stock"]),
    ({"filing_chunks": [], "errors": [{"recoverable": False, "message": "secret"}]}, "failed", ["filings"]),
    ({"filing_chunks": [], "news_articles": [], "include_filing_analysis": False, "include_news_sentiment": False}, "completed", []),
])
def test_job_and_result_share_truthful_terminal_status(client, monkeypatch, patch, expected, missing):
    async def run(ticker, company_name, **kwargs):
        state = await fake_run_agent(ticker, company_name)
        return {**state, **patch}
    monkeypatch.setattr(api_main, "run_agent", run)
    accepted = client.post("/analyze/async", json={"ticker": "AAPL"})
    response = client.get(accepted.headers["location"])
    body = response.json()
    assert body["status"] == body["result"]["status"] == expected
    assert body["result"]["missing"] == missing
    assert "secret" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_common_errors_are_typed_and_safe(client, monkeypatch):
    response = client.put("/analyze")
    assert set(response.json()["error"]) == {"code", "message", "error_id"}
    monkeypatch.setattr(api_main, "get_metrics", lambda: 1 / 0)
    with TestClient(api_main.app, raise_server_exceptions=False) as probe:
        response = probe.get("/metrics", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["error"]["code"] == "internal_error"


def test_auth_fail_closed_and_header_alternative(client, monkeypatch):
    client.headers.pop("Authorization")
    assert client.get("/metrics", headers={"X-API-Key": "test-secret"}).status_code == 200
    for path in ("/analyze", "/analyze/async", "/ingest"):
        assert client.post(path, json={"ticker": "AAPL"}).status_code == 401
    assert client.get("/jobs/00000000-0000-0000-0000-000000000000").status_code == 401
    monkeypatch.setattr(api_main.settings, "api_key", None)
    assert client.get("/stats", headers={"X-API-Key": "test-secret"}).status_code == 503


def test_ingestion_refresh_changes_existing_content_and_failures_are_safe(client, monkeypatch):
    contents = {"AAPL": 1}
    async def ingest(ticker, replace_existing=False):
        if replace_existing:
            contents[ticker] = 2
        return StubIngestResult()
    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", ingest)
    assert client.post("/ingest", json={"ticker": "AAPL"}).status_code == 200
    assert contents["AAPL"] == 1
    assert client.post("/ingest", json={"ticker": "AAPL", "force_refresh": True}).status_code == 200
    assert contents["AAPL"] == 2
    async def failed(ticker, replace_existing=False):
        result = StubIngestResult()
        result.success = False
        result.error = "secret provider details"
        return result
    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", failed)
    response = client.post("/ingest", json={"ticker": "AAPL"})
    assert response.status_code == 502
    assert "secret" not in response.text


# Trace: docs/test-plan.md §16, Ingest no 10-K.
def test_ingestion_returns_typed_missing_filing_reason(client, monkeypatch):
    from app.components.retrieval.ingestion import IngestionResult

    async def absent(ticker, replace_existing=False):
        return IngestionResult(
            ticker=ticker,
            filing_type="10-K",
            error=f"No 10-K filings found for {ticker}.",
            error_code="filing_not_found",
        )

    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", absent)
    response = client.post("/ingest", json={"ticker": "aapl"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "filing_not_found"
    assert response.json()["error"]["message"] == "No 10-K filings found for AAPL."


# Trace: docs/test-plan.md §16, analysis without filings and retrieval failure.
@pytest.mark.parametrize("retrieval_fails", [False, True])
@pytest.mark.parametrize("memo_text", ["# Investment Memo\n\nMarket evidence is limited.", ""])
def test_analysis_preserves_evidence_when_filings_are_unavailable(client, monkeypatch, retrieval_fails, memo_text):
    from types import SimpleNamespace

    from app.agents import graph
    from app.agents.state import apply_update, create_initial_state

    def search(*_args):
        if retrieval_fails:
            raise RuntimeError("private retrieval details")
        return SimpleNamespace(chunks=[])

    class MemoLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content=memo_text)

    monkeypatch.setattr(graph, "get_vector_store", lambda: object())
    monkeypatch.setattr(graph, "_search_filing_chunks", search)
    monkeypatch.setattr(graph, "get_llm", lambda _settings: MemoLLM())

    async def run(ticker, company_name, **_kwargs):
        state = create_initial_state(ticker, company_name)
        state["stock_data"] = {"ticker": ticker, "current_price": 123.0}
        state["news_articles"] = [{"title": "Market update", "snippet": "Shares rose.", "source": "Wire"}]
        state["sentiment_result"] = {"overall_sentiment": "positive", "positive_count": 1}
        for node in (graph.retrieve_sec_filings_node, graph.draft_memo_node, graph.verify_memo_node):
            state = apply_update(state, await node(state))
        return state

    monkeypatch.setattr(api_main, "run_agent", run)
    response = client.post("/analyze", json={"ticker": "AAPL"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == ("evidence_missing" if memo_text else "failed")
    assert payload["missing"] == ["filings"]
    assert payload["stock_data"]["current_price"] == 123.0
    assert len(payload["news_articles"]) == 1
    assert payload["sentiment"]["overall_sentiment"] == "positive"
    if memo_text:
        assert "SEC Filings: Not Available" in payload["investment_memo"]
    else:
        assert not payload["investment_memo"]
    assert all(item["source_type"] != "sec_filing" for item in payload["citations"])
    assert payload["verification"] is not None
    reason = next(error["message"] for error in payload["errors"] if error["step"] == "retrieve_filings")
    assert reason == ("Filing retrieval failed for AAPL." if retrieval_fails else "No indexed filings found for AAPL.")
    assert "private" not in response.text


def test_health_reports_unavailable_retrieval_and_configuration_only_bedrock(client, monkeypatch):
    response = client.get("/health")
    assert response.json()["components"]["chat_model"]["check"] == "configuration_only"
    def unavailable():
        raise RuntimeError("private backend location")
    monkeypatch.setattr(api_main, "get_vector_store", unavailable)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["components"]["vector_store"]["ok"] is False
    assert "private" not in response.text


def test_idempotent_replay_preserves_original_result(client, monkeypatch):
    first = client.post("/analyze/async", json={"ticker": "aapl"}, headers={"Idempotency-Key": "same"})
    original = client.get(first.headers["location"]).json()
    async def revised(ticker, company_name, **kwargs):
        state = await fake_run_agent(ticker, company_name)
        state["investment_memo"] = "New underlying evidence"
        return state
    monkeypatch.setattr(api_main, "run_agent", revised)
    replay = client.post("/analyze/async", json={"ticker": "AAPL"}, headers={"Idempotency-Key": "same"})
    assert replay.headers["location"] == first.headers["location"]
    assert client.get(replay.headers["location"]).json() == original
    fresh = client.post("/analyze/async", json={"ticker": "AAPL"}, headers={"Idempotency-Key": "new"})
    assert client.get(fresh.headers["location"]).json()["result"]["investment_memo"] == "New underlying evidence"
    assert "test-secret" not in api_main.run_store.path.read_text()


def test_rate_limit_expires_and_is_scoped_to_key(client, monkeypatch):
    from types import SimpleNamespace
    now = [1000.0]
    monkeypatch.setattr(api_main, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(api_main.settings, "api_rate_limit", 1)
    assert client.post("/analyze", json={"ticker": "AAPL"}).status_code == 200
    assert client.post("/ingest", json={"ticker": "AAPL"}).status_code == 429
    monkeypatch.setattr(api_main.settings, "api_key", "rotated-key")
    assert client.post("/analyze", json={"ticker": "AAPL"}, headers={"Authorization": "Bearer rotated-key"}).status_code == 200
    monkeypatch.setattr(api_main.settings, "api_key", "test-secret")
    now[0] += 61
    assert client.post("/ingest", json={"ticker": "AAPL"}).status_code == 200


def test_bedrock_does_not_depend_on_ollama_chat_model(client, monkeypatch):
    async def available(*, model=None, log_failure=False):
        return model == api_main.settings.ollama.embed_model
    monkeypatch.setattr(api_main, "check_ollama_health", available)
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(api_main.settings.llm, "aws_region", None)
    monkeypatch.setattr(api_main.settings.llm, "anthropic_api_key", None)
    monkeypatch.setattr(api_main.settings.llm, "openai_api_key", None)
    assert client.get("/health").status_code == 503


def test_openapi_defines_security_and_error_schema(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/analyze/async"]["post"]
    assert {"HTTPBearer": []} in operation["security"]
    assert {"APIKeyHeader": []} in operation["security"]
    assert operation["responses"]["401"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


# Trace: docs/test-plan.md §1 all-route OpenAPI contracts.
def test_all_nine_routes_document_actual_success_and_errors(client):
    schema = client.get("/openapi.json").json()
    routes = [("/", "get", "200"), ("/health", "get", "200"),
              ("/metrics", "get", "200"), ("/stats", "get", "200"),
              ("/analyze", "post", "200"), ("/analyze/async", "post", "202"),
              ("/jobs/{job_id}", "get", "200"), ("/ingest", "post", "200"),
              ("/ingest/{ticker}", "get", "200")]
    for path, method, status in routes:
        operation = schema["paths"][path][method]
        content = operation["responses"][status]["content"]
        if path == "/metrics":
            actual = client.get(path)
            assert actual.headers["content-type"] in content
            assert "application/json" not in content
        else:
            assert content["application/json"]["schema"].get("$ref")
        assert operation["responses"]["405"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


def test_job_poll_serves_recorded_progress_and_result_counts(client: TestClient):
    """Trace: docs/test-plan.md §18 (progress persisted and served; filing chunk count)."""
    job_id = client.post("/analyze/async", json={"ticker": "AAPL"}).json()["job_id"]
    polled = client.get(f"/jobs/{job_id}").json()
    assert polled["progress"] == []
    assert polled["result"]["filing_chunk_count"] == 1
    assert polled["result"]["usage"] is None

    step = {"node": "fetch_stock", "status": "completed", "started_at": "2026-10-07T12:00:00+00:00",
            "ended_at": "2026-10-07T12:00:01+00:00", "duration_ms": 1000.0}
    api_main.run_store.record_progress(job_id, [step, {"node": "draft_memo", "status": "running", "started_at": "2026-10-07T12:00:01+00:00"}])
    progress = client.get(f"/jobs/{job_id}").json()["progress"]
    assert progress[0] == step
    assert progress[1] == {"node": "draft_memo", "status": "running", "started_at": "2026-10-07T12:00:01+00:00", "ended_at": None, "duration_ms": None}
