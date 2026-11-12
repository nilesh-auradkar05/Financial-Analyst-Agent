"""Tests for API CORS config and dependency injection (items 2, 15).

Uses FastAPI's TestClient — no live server needed.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as api_main
from app.main import _get_store, app


class StatsOnlyStore:
    """A vector store that reports statistics for its named collection and holds no documents."""

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name

    def get_stats(self) -> dict[str, Any]:
        return {"collection_name": self.collection_name, "total_documents": 0}


@pytest.fixture
def mock_store():
    return StatsOnlyStore("test_collection")


@pytest.fixture
def client(mock_store, monkeypatch):
    """TestClient with vector store overridden via DI."""
    app.dependency_overrides[_get_store] = lambda: mock_store
    monkeypatch.setattr(api_main.settings, "api_key", "test-secret")
    test_client = TestClient(app, headers={"X-API-Key": "test-secret"})
    yield test_client
    app.dependency_overrides.clear()


class TestCORSConfig:
    """Verify CORS headers are set correctly (item 2)."""

    def test_allowed_origin_gets_cors_headers(self, client):
        resp = client.options(
            "/",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI returns 200 for preflight on allowed origins
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_disallowed_origin_no_cors_header(self, client):
        resp = client.get(
            "/",
            headers={"Origin": "https://evil-site.com"},
        )
        # The response should NOT have the evil origin echoed back
        acao = resp.headers.get("access-control-allow-origin", "")
        assert "evil-site.com" not in acao


class TestDependencyInjection:
    """Verify vector store is injected via Depends (item 15)."""

    def test_stats_uses_injected_store(self, client, mock_store):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "vector_store" in data
        assert data["vector_store"]["collection_name"] == "test_collection"

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Financial Analyst" in resp.json()["name"]
