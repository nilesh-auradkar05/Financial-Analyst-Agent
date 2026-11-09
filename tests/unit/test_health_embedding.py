"""Trace: docs/test-plan.md §1 REST hardening, the `GET /health` bullet (amended 2026-11-09, CM-6e
with CM-6a): health reports the configured embedding arm from configuration only.

The embedding endpoint is a loopback fake that counts what it receives; `/health` and startup must
send it nothing, because every live embedding call spends the 20 RPM quota. The chat plane is the
configured personal fallback and the vector store a stub, as the existing health tests arrange them.
No network beyond 127.0.0.1, dummy credentials.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.main as main
from app.config import settings
from tests.embedding_arms import (
    AZURE_SMALL,
    BEDROCK_COHERE,
    FOUNDRY_COHERE,
    Arm,
    use_embedding_settings,
)
from tests.unit.fake_embeddings import FakeClock, FakeEmbeddingProviders

OLLAMA_PORT = 11434


class _Store:
    count = 0


@pytest.fixture
def embed_server() -> Iterator[FakeEmbeddingProviders]:
    server = FakeEmbeddingProviders().start()
    yield server
    server.stop()


@pytest.fixture
def configure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embed_server: FakeEmbeddingProviders,
    fake_clock: FakeClock,
) -> Any:
    """Select an arm by environment, then make the rest of the service healthy around it.

    Time is the fake clock: a health check that wrongly embedded would be paced by sleeping, and
    must fail on the request count, not hang."""

    async def langsmith_status() -> dict[str, bool]:
        return {"connected": False}

    def choose(
        arm: Arm = FOUNDRY_COHERE, *, remove: tuple[str, ...] = (), **overrides: str
    ) -> None:
        env = {**arm.routed_to(embed_server.url), **overrides}
        for name in remove:
            env.pop(name)
        use_embedding_settings(monkeypatch, tmp_path, env)
        monkeypatch.setattr(settings.llm, "aws_region", None)
        monkeypatch.setattr(settings.llm, "anthropic_api_key", SecretStr("dummy"))
        monkeypatch.setattr(main, "check_langsmith_connection", langsmith_status)
        monkeypatch.setattr(main, "get_vector_store", lambda: _Store())

    return choose


def _health() -> Any:
    with TestClient(main.app) as client:
        return client.get("/health")


def test_health_reports_the_default_embedding_arm_from_configuration_only(
    configure: Any, embed_server: FakeEmbeddingProviders
) -> None:
    """test-plan §1 health: embedding component names provider and model; no embedding call."""
    configure()

    response = _health()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["components"]["embedding"] == {
        "ok": True,
        "provider": "foundry_cohere",
        "model": "cohere-embed-v5-fast",
        "check": "configuration_only",
    }
    assert embed_server.requests == []


def test_health_has_no_ollama_component(configure: Any) -> None:
    """test-plan §1 health: nothing in health depends on Ollama."""
    configure()

    response = _health()

    components = response.json()["components"]
    assert "ollama_embeddings" not in components
    assert not [name for name in components if "ollama" in name.lower()]
    assert "ollama" not in response.text.lower()


@pytest.mark.parametrize("setting", ["AZURE_API_KEY", "AZURE_OPENAI_ENDPOINT"])
@pytest.mark.parametrize("how", ["unset", "blank"])
def test_health_is_not_healthy_when_the_embedding_arm_lacks_its_settings(
    configure: Any, embed_server: FakeEmbeddingProviders, setting: str, how: str
) -> None:
    """test-plan §1 health: embedding component not ok when the arm's required settings are
    missing; 503 `degraded`; still no embedding call."""
    if how == "unset":
        configure(remove=(setting,))
    else:
        configure(**{setting: ""})

    response = _health()

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    embedding = payload["components"]["embedding"]
    assert embedding["ok"] is False
    assert embedding["provider"] == "foundry_cohere"
    assert embedding["model"] == "cohere-embed-v5-fast"
    assert embedding["check"] == "configuration_only"
    assert payload["components"]["vector_store"]["ok"] is True
    assert embed_server.requests == []


def test_health_reports_a_configured_azure_arm_as_ok(
    configure: Any, embed_server: FakeEmbeddingProviders
) -> None:
    """test-plan §1 health / §21 Provider selected by configuration: the component follows the
    configured arm, not a fixed one."""
    configure(AZURE_SMALL)

    response = _health()

    assert response.status_code == 200
    assert response.json()["components"]["embedding"] == {
        "ok": True,
        "provider": "azure_openai",
        "model": AZURE_SMALL.model,
        "check": "configuration_only",
    }
    assert embed_server.requests == []


def test_health_names_a_bedrock_arm_without_calling_it(
    configure: Any, embed_server: FakeEmbeddingProviders
) -> None:
    """test-plan §1 health / §21 Provider selected by configuration."""
    configure(BEDROCK_COHERE)

    embedding = _health().json()["components"]["embedding"]

    assert (embedding["provider"], embedding["model"]) == ("bedrock", BEDROCK_COHERE.model)
    assert embedding["check"] == "configuration_only"
    assert embed_server.requests == []


def test_startup_and_health_never_contact_a_local_model_server(
    monkeypatch: pytest.MonkeyPatch, configure: Any, embed_server: FakeEmbeddingProviders
) -> None:
    """test-plan §1 health: startup performs no Ollama check (no connection to its default port)
    and no embedding call."""
    configure()
    connected: list[Any] = []
    real_connect = socket.socket.connect

    def recording_connect(self: socket.socket, address: Any) -> None:
        connected.append(address)
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", recording_connect)

    with TestClient(main.app) as client:
        client.get("/health")
        client.get("/health")

    ports = [address[1] for address in connected if isinstance(address, tuple)]
    assert OLLAMA_PORT not in ports
    assert embed_server.requests == []


def test_the_service_module_no_longer_exposes_an_ollama_probe() -> None:
    """test-plan §1 health / §25: the probe is gone from the service entry point."""
    assert not hasattr(main, "check_ollama_health")
