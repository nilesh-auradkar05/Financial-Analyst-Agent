"""Public model construction and health behavior: docs/test-plan.md §§1, 8."""

import httpx
import pytest

from app.config import Settings, settings
from app.services.llm import ANALYST_SYSTEM_PROMPT, check_ollama_health, get_llm


def test_local_model_uses_explicit_configuration():
    config = Settings(_env_file=None)
    config.llm.provider = "ollama"
    config.ollama.llm_model = "test-local"
    config.llm.temperature = 0.2
    model = get_llm(config)
    assert model.model == "test-local"
    assert model.temperature == 0.2


def test_system_prompt_requires_citations():
    assert "[N]" in ANALYST_SYSTEM_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize("models, status, expected", [
    (["embedding-test", "chat-test"], 200, True),
    (["chat-test-other"], 200, False),
    (["chat-test"], 503, False),
    ([], 200, False),
])
async def test_model_health_uses_exact_available_model(monkeypatch, models, status, expected):
    monkeypatch.setattr(settings.ollama, "llm_model", "chat-test")
    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"models": [{"name": name} for name in models]}))
    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", lambda: client_type(transport=transport))
    assert await check_ollama_health() is expected


@pytest.mark.asyncio
async def test_embedding_health_does_not_require_chat_model(monkeypatch):
    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"name": "embedding-test"}]}))
    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", lambda: client_type(transport=transport))
    assert await check_ollama_health(model="embedding-test") is True
    assert await check_ollama_health(model="missing-chat") is False


@pytest.mark.asyncio
async def test_unreachable_model_server_is_unhealthy(monkeypatch):
    client_type = httpx.AsyncClient
    def unavailable(request):
        raise httpx.ConnectError("unavailable", request=request)
    transport = httpx.MockTransport(unavailable)
    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", lambda: client_type(transport=transport))
    assert await check_ollama_health() is False
