"""Tests for app.services.llm — replaces the old ``_main()`` smoke test.

Mocks the Ollama server; no GPU needed for CI.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.services.llm import (
    ANALYST_SYSTEM_PROMPT,
    MEMO_TEMPLATE,
    check_ollama_health,
    get_llm,
)


class TestGetLLM:
    def test_delegates_to_provider(self):
        sentinel = object()
        with patch("app.services.llm.get_provider_llm", return_value=sentinel) as mock_get:
            llm = get_llm()
        assert llm is sentinel
        mock_get.assert_called_once()

    def test_accepts_explicit_settings(self):
        sentinel = object()
        with patch("app.services.llm.get_provider_llm", return_value=sentinel) as mock_get:
            llm = get_llm(settings)
        assert llm is sentinel
        mock_get.assert_called_once_with(settings)


class TestPrompts:
    def test_system_prompt_has_citation_guidance(self):
        assert "[N]" in ANALYST_SYSTEM_PROMPT

    def test_memo_template_has_placeholders(self):
        assert "{company_name}" in MEMO_TEMPLATE
        assert "{ticker}" in MEMO_TEMPLATE
        assert "{context}" in MEMO_TEMPLATE


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_server(self):
        # httpx.Response.json() is SYNC — use MagicMock for it
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3.5:9b"}],
        }

        with patch("app.services.llm.httpx.AsyncClient") as mockclient:
            # The key: __aenter__ must return the SAME instance
            # we configure, otherwise async-with creates a new one
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__.return_value = instance
            mockclient.return_value = instance

            result = await check_ollama_health()
            # Behavior: a 200 response listing the configured model → healthy.
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_server(self):
        with patch("app.services.llm.httpx.AsyncClient") as mockclient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=Exception("connection refused"))
            instance.__aenter__.return_value = instance
            mockclient.return_value = instance

            result = await check_ollama_health()
            assert result is False

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama2:7b"}],
        }

        with patch("app.services.llm.httpx.AsyncClient") as mockclient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__.return_value = instance
            mockclient.return_value = instance

            result = await check_ollama_health()
            assert result is False
