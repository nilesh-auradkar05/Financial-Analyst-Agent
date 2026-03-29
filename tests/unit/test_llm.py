"""Tests for models.llm — replaces the old ``_main()`` smoke test.

Mocks the Ollama server; no GPU needed for CI.
"""

from unittest.mock import AsyncMock, patch

import pytest

from models.llm import (
    ANALYST_SYSTEM_PROMPT,
    MEMO_TEMPLATE,
    check_ollama_health,
    get_llm,
)


class TestGetLLM:
    def test_returns_chat_ollama(self):
        llm = get_llm()
        assert llm.model is not None

    def test_temperature_override(self):
        llm = get_llm(temperature=0.3)
        assert llm.temperature == 0.3


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
            "models": [{"name": "qwen3-vl:8b"}],
        }

        with patch("models.llm.httpx.AsyncClient") as MockClient:
            # The key: __aenter__ must return the SAME instance
            # we configure, otherwise async-with creates a new one
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__.return_value = instance
            MockClient.return_value = instance

            result = await check_ollama_health()
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_server(self):
        with patch("models.llm.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=Exception("connection refused"))
            instance.__aenter__.return_value = instance
            MockClient.return_value = instance

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

        with patch("models.llm.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__.return_value = instance
            MockClient.return_value = instance

            result = await check_ollama_health()
            assert result is False