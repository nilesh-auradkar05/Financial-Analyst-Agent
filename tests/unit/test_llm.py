"""Tests for app.services.llm.

Public surface: ``get_llm``, ``check_ollama_health`` and ``ANALYST_SYSTEM_PROMPT``.
The Ollama HTTP server is the only external boundary; it is faked with a real
``httpx.MockTransport`` and assertions stay on returned values. No GPU or server needed.
"""

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings
from app.services.llm import ANALYST_SYSTEM_PROMPT, check_ollama_health, get_llm

_RealAsyncClient = httpx.AsyncClient


@pytest.fixture
def ollama_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Callable[[httpx.Request], httpx.Response]], None]]:
    """Install a fake Ollama server behind ``httpx.AsyncClient`` (the external boundary)."""

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            return _RealAsyncClient(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    yield install


def _tags_server(model_names: list[str], status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    """Faithful /api/tags fake: answers only that route, from the given model list."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/tags":
            return httpx.Response(404)
        return httpx.Response(status, json={"models": [{"name": n} for n in model_names]})

    return handler


class TestGetLLM:
    """test-plan §8 Config: "Typed settings load" / provider selection via typed settings."""

    def test_ollama_provider_returns_chat_model(self) -> None:
        config = settings.model_copy(deep=True)
        config.llm.provider = "ollama"
        assert isinstance(get_llm(config), BaseChatModel)

    def test_unsupported_provider_fails_fast(self) -> None:
        config = settings.model_copy(deep=True)
        object.__setattr__(config.llm, "provider", "nonsense")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(config)


class TestAnalystPrompt:
    """test-plan §7 RAG/memo: "Citation mapping" and "Missing evidence" (prompt-level contract)."""

    def test_requires_numbered_citations(self) -> None:
        assert "[N]" in ANALYST_SYSTEM_PROMPT or "[1]" in ANALYST_SYSTEM_PROMPT

    def test_forbids_unsupported_facts_and_invented_citations(self) -> None:
        text = ANALYST_SYSTEM_PROMPT.lower()
        assert "never invent citation numbers" in text
        assert "unsupported" in text


class TestHealthCheck:
    """test-plan §1 API endpoints: ``GET /health`` returns service health (Ollama probe)."""

    @pytest.mark.asyncio
    async def test_healthy_when_configured_model_listed(self, ollama_server: Any) -> None:
        ollama_server(_tags_server([settings.ollama.llm_model]))
        assert await check_ollama_health() is True

    @pytest.mark.asyncio
    async def test_unhealthy_when_model_missing(self, ollama_server: Any) -> None:
        ollama_server(_tags_server(["some-other-model:1b"]))
        assert await check_ollama_health() is False

    @pytest.mark.asyncio
    async def test_unhealthy_on_http_error(self, ollama_server: Any) -> None:
        ollama_server(_tags_server([settings.ollama.llm_model], status=500))
        assert await check_ollama_health() is False

    @pytest.mark.asyncio
    async def test_unhealthy_when_server_unreachable(self, ollama_server: Any) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        ollama_server(refuse)
        assert await check_ollama_health() is False
