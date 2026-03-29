"""Tests for API CORS config and dependency injection (items 2, 15).

Uses FastAPI's TestClient — no live server needed.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import _get_store, app
from rag.vector_store import ChromaDBVectorStore


@pytest.fixture
def mock_store():
    """In-memory ChromaDB store for testing."""
    return ChromaDBVectorStore(
        collection_name="test_collection",
        persist_directory=None,  # in-memory
    )


@pytest.fixture
def client(mock_store):
    """TestClient with vector store overridden via DI."""
    app.dependency_overrides[_get_store] = lambda: mock_store
    yield TestClient(app)
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