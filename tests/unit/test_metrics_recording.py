"""Trace: docs/test-plan.md §1 rows "Async submission counted" and "Errors counted by component"."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import app.main as api
from app.services.run_store import FileBackedRunStore

AUTH = {"X-API-Key": "private-auth-value"}
COMPONENTS = ("analysis_job", "analysis", "ingestion", "api")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "run_store", FileBackedRunStore(tmp_path / "runs.json"))
    monkeypatch.setattr(api, "submission_limiter", api.SubmissionRateLimiter())
    monkeypatch.setattr(api.settings, "api_key", "private-auth-value")
    monkeypatch.setattr(api.settings.langsmith, "tracing_v2", False)
    monkeypatch.setattr(api.settings.langsmith, "api_key", None)
    monkeypatch.setattr(api.settings.finbert, "preload", False)
    with TestClient(api.app) as test_client:
        yield test_client


def metric(client: TestClient, name: str, **labels: str) -> float:
    """Sum of the exported samples called `name` that carry `labels`."""
    text = client.get("/metrics", headers=AUTH).text
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


def errors(client: TestClient) -> dict[str, float]:
    return {
        component: metric(client, "alpha_analyst_errors_total", component=component)
        for component in COMPONENTS
    }


def failing_agent(monkeypatch):
    async def failure(*args, **kwargs):
        raise RuntimeError("agent crashed")

    monkeypatch.setattr(api, "run_agent", failure)


def test_async_submission_is_counted_and_raises_no_error(client, monkeypatch, sample_agent_state):
    async def agent(*args, **kwargs):
        return sample_agent_state

    monkeypatch.setattr(api, "run_agent", agent)
    endpoint = {"endpoint": "/analyze/async"}
    requests_before = metric(client, "alpha_analyst_requests_total", **endpoint)
    latency_before = metric(client, "alpha_analyst_request_latency_seconds_count", **endpoint)
    errors_before = errors(client)

    accepted = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=AUTH)

    assert accepted.status_code == 202
    assert client.get(accepted.headers["Location"], headers=AUTH).json()["status"] != "failed"
    assert metric(client, "alpha_analyst_requests_total", **endpoint) == requests_before + 1
    assert (
        metric(client, "alpha_analyst_request_latency_seconds_count", **endpoint)
        == latency_before + 1
    )
    assert errors(client) == errors_before


def test_crashed_async_job_counts_one_analysis_job_error(client, monkeypatch):
    failing_agent(monkeypatch)
    before = errors(client)

    accepted = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=AUTH)

    assert client.get(accepted.headers["Location"], headers=AUTH).json()["status"] == "failed"
    assert errors(client) == {**before, "analysis_job": before["analysis_job"] + 1}


def test_crashed_sync_analysis_counts_one_analysis_error(client, monkeypatch):
    failing_agent(monkeypatch)
    before = errors(client)

    assert client.post("/analyze", json={"ticker": "AAPL"}, headers=AUTH).status_code == 500
    assert errors(client) == {**before, "analysis": before["analysis"] + 1}


@pytest.mark.parametrize("outcome", ["raises", "reports_failure"])
def test_failed_ingestion_counts_one_ingestion_error(client, monkeypatch, outcome):
    async def ingest(ticker, **kwargs):
        if outcome == "raises":
            raise RuntimeError("store unavailable")
        return SimpleNamespace(
            success=False, error_code="embedding_failed", sections_processed=None
        )

    monkeypatch.setattr(api, "ingest_10k_for_ticker", ingest)
    before = errors(client)

    assert client.post("/ingest", json={"ticker": "AAPL"}, headers=AUTH).status_code == 502
    assert errors(client) == {**before, "ingestion": before["ingestion"] + 1}


def test_missing_filing_is_not_an_ingestion_error(client, monkeypatch):
    async def ingest(ticker, **kwargs):
        return SimpleNamespace(
            success=False, error_code="filing_not_found", sections_processed=None
        )

    monkeypatch.setattr(api, "ingest_10k_for_ticker", ingest)
    before = errors(client)

    assert client.post("/ingest", json={"ticker": "GSPC"}, headers=AUTH).status_code == 404
    assert errors(client) == before


def test_unhandled_exception_counts_one_api_error(client, monkeypatch):
    before = errors(client)
    with monkeypatch.context() as broken:
        broken.setattr(api, "get_metrics", lambda: 1 / 0)
        assert client.get("/metrics", headers=AUTH).status_code == 500

    assert errors(client) == {**before, "api": before["api"] + 1}
