"""Trace: docs/test-plan.md §1 maintained live client contract."""
import sys

import httpx
import pytest

from scripts import smoke_test_live_pipeline as smoke


@pytest.mark.parametrize("terminal", ["completed", "degraded", "evidence_missing", "failed"])
def test_async_only_smoke_authenticates_and_reports_terminal_outcomes(monkeypatch, capsys, terminal):
    monkeypatch.setenv("API_KEY", "private-test-key")
    monkeypatch.setattr(sys, "argv", ["smoke", "--async-only", "--skip-ingest"])
    def respond(request):
        if request.headers.get("Authorization") != "Bearer private-test-key":
            return httpx.Response(401)
        if request.url.path == "/analyze":
            return httpx.Response(400)
        if request.url.path == "/analyze/async":
            return httpx.Response(202, json={"job_id": "job-1", "status": "pending"})
        if request.url.path == "/jobs/job-1":
            return httpx.Response(200, json={"job_id": "job-1", "status": terminal})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="# metrics")
        return httpx.Response(200, json={"status": "healthy"})
    client_type = httpx.Client
    monkeypatch.setattr(smoke.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    if terminal == "completed":
        smoke.main()
    else:
        with pytest.raises(SystemExit) as error:
            smoke.main()
        assert error.value.code == 1
    output = capsys.readouterr().out
    assert terminal in output
    assert "private-test-key" not in output
