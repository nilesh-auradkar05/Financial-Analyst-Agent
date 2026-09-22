"""Trace: docs/test-plan.md §1 Correlated tracing (T2-01 through T2-07)."""
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.config import LangSmithSettings


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_tracing_false_is_boolean(value, monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    config = LangSmithSettings(_env_file=None, LANGCHAIN_TRACING_V2=value)
    assert config.tracing_v2 is False


def test_request_ids_cover_errors_and_invalid_values(offline):
    with TestClient(api.app, raise_server_exceptions=False) as client:
        request_id = str(uuid4())
        response = client.get("/missing", headers={"X-Request-ID": request_id})
        assert response.status_code == 404
        assert response.headers["X-Request-ID"] == request_id
        response = client.get("/", headers={"X-Request-ID": "bad identifier"})
        assert response.status_code == 400
        assert str(UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]
        assert "X-Trace-ID" not in response.headers


class RunCollector:
    otel_exporter = None
    """Faithful public SDK create/update transport: persisted runs keyed by UUID."""
    def __init__(self):
        self.runs = {}
        self.flushed = False

    def create_run(self, **payload):
        self.runs[str(payload["id"])] = dict(payload)

    def update_run(self, run_id, **payload):
        self.runs.setdefault(str(run_id), {"id": run_id}).update(payload)

    def flush(self, timeout=None):
        self.flushed = True


@pytest.fixture
def offline(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import app.observability.langsmith as tracing
    from app.services.run_store import FileBackedRunStore
    async def healthy(**kwargs):
        return True
    monkeypatch.setattr(api, "check_ollama_health", healthy)
    monkeypatch.setattr(api, "get_vector_store", lambda: SimpleNamespace(count=0))
    monkeypatch.setattr(api, "run_store", FileBackedRunStore(tmp_path / "runs.json"))
    monkeypatch.setattr(api, "submission_limiter", api.SubmissionRateLimiter())
    monkeypatch.setattr(api.settings, "api_key", "private-auth-value")
    monkeypatch.setattr(api.settings.langsmith, "tracing_v2", False)
    monkeypatch.setattr(api.settings.langsmith, "api_key", None)
    import requests
    def deny_network(*args, **kwargs):
        raise AssertionError("Offline tracing tests must never use network")
    monkeypatch.setattr(requests.Session, "request", deny_network)
    collector = RunCollector()
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: collector)
    return collector


@pytest.fixture
def traced(offline, monkeypatch):
    monkeypatch.setattr(api.settings.langsmith, "tracing_v2", True)
    monkeypatch.setattr(api.settings.langsmith, "api_key", "private-export-key")
    return offline


def test_http_trace_headers_and_metadata(traced):
    with TestClient(api.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    trace_id = response.headers["X-Trace-ID"]
    runs = list(traced.runs.values())
    assert len(runs) == 1
    root = runs[0]
    assert str(root["trace_id"]) == trace_id
    assert root["name"] == "HTTP GET /"
    metadata = root["extra"]["metadata"]
    assert metadata["request_id"] == response.headers["X-Request-ID"]
    assert metadata["status_code"] == 200
    assert metadata["route"] == "/"
    assert metadata["duration_ms"] >= 0
    assert traced.flushed



@pytest.fixture
def evidence(monkeypatch):
    import asyncio
    import re
    from types import SimpleNamespace

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langsmith import traceable

    from app.agents import graph

    class EvidenceChat(BaseChatModel):
        model: str = "offline-evidence-model"
        temperature: float = 0.0

        @property
        def _llm_type(self):
            return "offline-evidence"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompt = str(messages[-1].content)
            sources = re.findall(r"\[(\d+)\] [^\n]+\n([^\n]+)", prompt)
            memo = "## Executive Summary\n" + "\n".join(
                f"{text} [{index}]" for index, text in sources
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=[{"type": "reasoning", "reasoning": "provider-visible-reasoning"},
                         {"type": "text", "text": memo}],
                usage_metadata={"input_tokens": len(prompt.split()), "output_tokens": len(memo.split()),
                                "total_tokens": len(prompt.split()) + len(memo.split())},
            ))])

    @traceable(name="get_stock_data", run_type="tool")
    async def stock(ticker):
        await asyncio.sleep(0.02)
        return SimpleNamespace(ticker=ticker, company_name=ticker + " Corporation",
                               current_price=float(sum(map(ord, ticker))), price_change_percent=1.0,
                               market_cap=1000000, pe_ratio=10.0, fifty_two_week_high=400.0,
                               fifty_two_week_low=50.0, volume=100, dividend_yield=0.0,
                               sector="Technology", industry="Software", recommendation="hold")

    @traceable(name="search_company_news", run_type="tool")
    async def news(query, max_results=10):
        await asyncio.sleep(0)
        return [SimpleNamespace(title=query, url="https://example.test/news", source="Local evidence",
                                snippet=query + " reports software revenue growth.", published_date="2026-09-01")][:max_results]

    class Store:
        def search_by_ticker(self, query, ticker, n_results=3):
            return SimpleNamespace(chunks=[SimpleNamespace(text=f"{ticker} sells software and faces competition.",
                section="Business", metadata={"filing_type": "10-K"}, filing_date="2026-01-01", relevance_score=1.0)])

    monkeypatch.setattr(graph, "get_stock_data", stock)
    monkeypatch.setattr(graph, "search_company_news", news)
    monkeypatch.setattr(graph, "get_vector_store", Store)
    monkeypatch.setattr(graph, "analyze_sentiment_batch", lambda texts: [SimpleNamespace(label="positive") for text in texts])
    monkeypatch.setattr(graph, "get_llm", lambda settings: EvidenceChat())


def descendants(runs, root_id):
    found = set()
    pending = [root_id]
    while pending:
        parent = pending.pop()
        for run in runs:
            run_id = str(run["id"])
            if str(run.get("parent_run_id")) == parent and run_id not in found:
                found.add(run_id)
                pending.append(run_id)
    return [run for run in runs if str(run["id"]) in found]


def test_real_graph_has_native_model_tools_verifier_and_text_only_memo(traced, evidence):
    with TestClient(api.app) as client:
        response = client.post("/analyze", json={"ticker": "AAPL"}, headers={
            "Authorization": "Bearer private-auth-value", "Cookie": "private-cookie-value"})
    assert response.status_code == 200
    assert "provider-visible-reasoning" not in response.text
    assert "AAPL" in response.json()["investment_memo"]
    runs = list(traced.runs.values())
    roots = [run for run in runs if not run.get("parent_run_id")]
    assert len(roots) == 1
    children = descendants(runs, str(roots[0]["id"]))
    assert len(children) == len(runs) - 1
    names = [run["name"] for run in children]
    assert names.count("run_financial_analysis") == 1
    assert names.count("financial_analyst_graph") == 1
    assert {"get_stock_data", "search_company_news", "retrieve_filings", "verify_memo"} <= set(names)
    assert all(run["run_type"] == "chain" for run in children if run["name"] == "draft_memo")
    llms = [run for run in children if run["run_type"] == "llm"]
    assert len(llms) == 1
    assert "provider-visible-reasoning" in str(llms[0]["outputs"])
    assert "usage_metadata" in str(llms[0]["outputs"])
    assert llms[0]["inputs"]
    assert "available_tools" in roots[0]["extra"]["metadata"]
    assert "tools" not in roots[0]["extra"]["metadata"]
    draft = next(run for run in children if run["name"] == "draft_memo" and run["extra"]["metadata"].get("model") == "offline-evidence-model")
    assert draft["extra"]["metadata"]["temperature"] == 0.0
    exported = str(runs)
    assert all(secret not in exported for secret in ("private-auth-value", "private-cookie-value", "private-export-key"))


def test_nested_app_failure_exports_only_safe_error(traced, evidence, monkeypatch):
    from app.agents import graph
    from app.observability.langsmith import app_traceable

    @app_traceable(name="analyze_sentiment_batch", run_type="chain", tags=["sentiment"])
    def failed_sentiment(_texts):
        raise RuntimeError("private-nested-exception-secret")

    monkeypatch.setattr(graph, "analyze_sentiment_batch", failed_sentiment)
    with TestClient(api.app) as client:
        response = client.post(
            "/analyze",
            json={"ticker": "AAPL"},
            headers={"X-API-Key": "private-auth-value"},
        )

    assert response.status_code == 200
    assert "private-nested-exception-secret" not in str(traced.runs)
    nested = next(run for run in traced.runs.values() if run["name"] == "analyze_sentiment_batch")
    assert nested.get("error") is None
    assert nested["extra"]["metadata"]["outcome"] == "failed"
    assert nested["extra"]["metadata"]["error_code"] == "app_stage_failed"


def test_verifier_failure_never_exports_raw_error(traced, evidence, monkeypatch):
    from app.agents import graph

    def failed_verifier(*_args, **_kwargs):
        raise RuntimeError("private-verifier-exception-secret")

    monkeypatch.setattr(graph, "evaluate_memo_grounding", failed_verifier)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/analyze",
            json={"ticker": "AAPL"},
            headers={"X-API-Key": "private-auth-value"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["verification"]["passed"] is False
    assert any(error["step"] == "verify_memo" for error in response.json()["errors"])
    assert "private-verifier-exception-secret" not in str(traced.runs)
    assert any(
        "verification_failed" in str(run.get("outputs"))
        for run in traced.runs.values()
        if run["name"] == "verify_memo"
    )


def test_async_original_correlation_survives_parent_end_replay_and_poll(traced, evidence):
    request_id = str(uuid4())
    headers = {"X-API-Key": "private-auth-value", "X-Request-ID": request_id, "Idempotency-Key": "stable-job"}
    with TestClient(api.app) as client:
        first = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=headers)
        poll = client.get(first.headers["Location"], headers={"X-API-Key": "private-auth-value"})
        replay = client.post("/analyze/async", json={"ticker": "AAPL"}, headers={**headers, "X-Request-ID": str(uuid4())})
    assert first.status_code == replay.status_code == 202
    original = first.json()
    assert original["request_id"] == request_id
    assert original["trace_id"] == first.headers["X-Trace-ID"]
    for body in (poll.json(), poll.json()["result"], replay.json()):
        assert body["request_id"] == request_id
        assert body["trace_id"] == original["trace_id"]
    assert poll.headers["X-Request-ID"] != request_id
    assert poll.headers["X-Trace-ID"] != original["trace_id"]
    assert replay.headers["X-Trace-ID"] != original["trace_id"]
    runs = list(traced.runs.values())
    jobs = [run for run in runs if run["name"] == "analysis_job"]
    assert len(jobs) == 1
    root = next(run for run in runs if str(run["id"]) == str(jobs[0]["parent_run_id"]))
    assert str(jobs[0]["trace_id"]) == original["trace_id"]
    assert root["end_time"] < jobs[0]["end_time"]
    assert root["outputs"]["status_code"] == 202
    assert len([run for run in descendants(runs, str(jobs[0]["id"])) if run["name"] == "financial_analyst_graph"]) == 1


@pytest.mark.asyncio
async def test_concurrent_requests_never_cross_trace_or_ticker(traced, evidence):
    import asyncio

    import httpx
    async with api.lifespan(api.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            ids = [str(uuid4()), str(uuid4())]
            responses = await asyncio.gather(*[
                client.post("/analyze", json={"ticker": ticker}, headers={"X-API-Key": "private-auth-value", "X-Request-ID": request_id})
                for ticker, request_id in zip(("AAPL", "MSFT"), ids, strict=True)
            ])
    runs = list(traced.runs.values())
    for ticker, request_id, response in zip(("AAPL", "MSFT"), ids, responses, strict=True):
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == response.json()["request_id"] == request_id
        root = next(run for run in runs if str(run["id"]) == response.headers["X-Trace-ID"])
        agent = next(run for run in descendants(runs, str(root["id"])) if run["name"] == "run_financial_analysis")
        assert agent["inputs"]["ticker"] == ticker
        assert all(str(run["trace_id"]) == response.headers["X-Trace-ID"] for run in descendants(runs, str(root["id"])))


@pytest.mark.asyncio
async def test_native_carrier_restores_job_after_http_context_closes(traced, evidence):
    from app.observability.langsmith import trace_boundary
    async with api.lifespan(api.app):
        async with trace_boundary("HTTP accepted", metadata={}) as root:
            headers = root.to_headers()
            root.end(outputs={"status_code": 202})
            parent_id, trace_id = str(root.id), str(root.trace_id)
        async with trace_boundary("analysis_job", metadata={}, parent=headers) as job:
            result = await api.run_agent("AAPL")
            job.end(outputs={"ticker": result["ticker"]})
    jobs = [run for run in traced.runs.values() if run["name"] == "analysis_job"]
    assert len(jobs) == 1
    assert str(jobs[0]["parent_run_id"]) == parent_id
    assert str(jobs[0]["trace_id"]) == trace_id
    assert traced.runs[parent_id]["end_time"] <= jobs[0]["start_time"]
    assert any(run["run_type"] == "llm" for run in descendants(list(traced.runs.values()), str(jobs[0]["id"])))


@pytest.mark.parametrize("enabled,key", [(False, "unused-key"), (True, None), (True, "   ")])
def test_disabled_or_unconfigured_never_exports(offline, monkeypatch, enabled, key):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(api.settings.langsmith, "tracing_v2", enabled)
    monkeypatch.setattr(api.settings.langsmith, "api_key", key)
    with TestClient(api.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "X-Trace-ID" not in response.headers
    assert not offline.runs
    assert not offline.flushed


@pytest.mark.parametrize("prefix", ["LANGSMITH", "LANGCHAIN"])
def test_configuration_aliases_export_one_root(offline, monkeypatch, prefix):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    key = "LANGSMITH_TRACING" if prefix == "LANGSMITH" else "LANGCHAIN_TRACING_V2"
    config = LangSmithSettings(_env_file=None, **{key: "true", prefix + "_API_KEY": "offline", prefix + "_PROJECT": "offline-project"})
    monkeypatch.setattr(api.settings, "langsmith", config)
    with TestClient(api.app) as client:
        response = client.get("/")
    assert len(offline.runs) == 1
    assert response.headers["X-Trace-ID"] in offline.runs
    assert next(iter(offline.runs.values()))["session_name"] == "offline-project"


def test_safe_error_correlation_and_failed_job(traced, monkeypatch):
    async def failure(*args, **kwargs):
        raise RuntimeError("private-exception-secret")
    monkeypatch.setattr(api, "run_agent", failure)
    monkeypatch.setattr(api, "get_metrics", lambda: 1 / 0)
    headers = {"X-API-Key": "private-auth-value", "Cookie": "private-cookie-value"}
    with TestClient(api.app) as client:
        responses = [client.get("/missing"), client.post("/analyze", json={}, headers=headers),
                     client.get("/metrics", headers=headers), client.post("/analyze", json={"ticker": "AAPL"}, headers=headers)]
        accepted = client.post("/analyze/async", json={"ticker": "AAPL"}, headers=headers)
        poll = client.get(accepted.headers["Location"], headers=headers)
    assert [r.status_code for r in responses] == [404, 422, 500, 500]
    for response in responses:
        assert UUID(response.headers["X-Request-ID"])
        metadata = traced.runs[response.headers["X-Trace-ID"]]["extra"]["metadata"]
        assert metadata["code"] == response.json()["error"]["code"]
        assert metadata["error_id"] == response.json()["error"]["error_id"]
    assert poll.json()["status"] == "failed"
    assert poll.json()["request_id"] == accepted.json()["request_id"]
    assert poll.json()["trace_id"] == accepted.headers["X-Trace-ID"]
    assert all(value not in str(traced.runs) for value in ("private-exception-secret", "private-auth-value", "private-cookie-value", "private-export-key"))
    job = next(run for run in traced.runs.values() if run["name"] == "analysis_job")
    assert job["error"] == "analysis_failed"
    assert job["outputs"]["error_id"] in poll.json()["error"]


@pytest.mark.parametrize("phase", ["initialize", "create", "update", "flush"])
def test_exporter_fault_preserves_success(traced, monkeypatch, phase):
    import app.observability.langsmith as tracing
    def failed(*args, **kwargs):
        raise RuntimeError("private-exporter-secret")
    if phase == "initialize":
        monkeypatch.setattr(tracing, "Client", failed)
    else:
        monkeypatch.setattr(traced, {"create": "create_run", "update": "update_run", "flush": "flush"}[phase], failed)
    with TestClient(api.app) as client:
        assert client.get("/").status_code == 200


@pytest.mark.asyncio
async def test_shutdown_flush_is_bounded(traced, monkeypatch):
    import asyncio
    import threading

    import app.observability.langsmith as tracing
    released = threading.Event()
    def stalled(timeout=None):
        released.wait(1)
    monkeypatch.setattr(traced, "flush", stalled)
    monkeypatch.setattr(tracing, "FLUSH_TIMEOUT_SECONDS", 0.01)
    tracing.setup_langsmith_env()
    try:
        await asyncio.wait_for(tracing.shutdown_langsmith(), timeout=0.5)
        assert tracing.get_client() is None
    finally:
        released.set()


def test_modern_alias_wins_and_default_is_off(monkeypatch):
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    assert LangSmithSettings(_env_file=None).tracing_v2 is False
    assert LangSmithSettings(_env_file=None, LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="true").tracing_v2 is False
