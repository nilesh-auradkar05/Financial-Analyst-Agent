"""Tests for app.config — covers defaults, custom values, validation, CORS, retry."""

import pytest
from pydantic import ValidationError

from app.config import (
    LLMSettings,
    OllamaSettings,
    RetrySettings,
    Settings,
    settings,
    validate_settings,
)


class TestSettingsDefaults:
    """Easy — verify defaults load correctly."""

    def test_settings_singleton_exists(self):
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_ollama_defaults(self):
        assert settings.ollama.base_url == "http://localhost:11434"
        assert settings.ollama.llm_model == "qwen3.5:9b"
        assert settings.ollama.embed_model == "qwen3-embedding:4b"

    def test_chroma_defaults(self):
        assert settings.chroma.collection_name == "sec_filings"
        assert "chroma" in settings.chroma.persist_dir

    def test_cors_defaults(self):
        assert isinstance(settings.cors_allow_origins, list)
        assert len(settings.cors_allow_origins) >= 1
        assert all(o.startswith("http") for o in settings.cors_allow_origins)

    def test_retry_defaults(self):
        assert settings.retry.max_attempts >= 1
        assert settings.retry.min_wait_seconds > 0
        assert settings.retry.max_wait_seconds >= settings.retry.min_wait_seconds

    def test_llm_timeout_default(self):
        assert settings.llm.request_timeout_seconds > 0


class TestSettingsCustomValues:
    """Medium — verify custom instantiation."""

    def test_ollama_custom_values(self):
        custom = OllamaSettings(
            base_url="http://custom:11434",
            llm_model="llama2",
            temperature=0.5,
        )
        assert custom.base_url == "http://custom:11434"
        assert custom.llm_model == "llama2"
        assert custom.temperature == 0.5

    def test_retry_custom_values(self):
        custom = RetrySettings(
            max_attempts=5,
            min_wait_seconds=2.0,
            max_wait_seconds=30.0,
            http_timeout_seconds=60.0,
        )
        assert custom.max_attempts == 5
        assert custom.http_timeout_seconds == 60.0

    def test_llm_custom_timeout(self):
        custom = LLMSettings(request_timeout_seconds=45.0)
        assert custom.request_timeout_seconds == 45.0


class TestValidation:
    """Hard — validate_settings edge cases."""

    def test_validate_returns_list(self):
        warnings = validate_settings()
        assert isinstance(warnings, list)

    def test_missing_api_keys_produce_warnings(self, monkeypatch):
        """When Tavily/LangSmith keys are absent, validate_settings names them.

        Forces the absent-key condition on the real settings object and asserts
        the real validate_settings() output — not just that it returns a list.
        """
        monkeypatch.setattr(settings.tavily, "api_key", None)
        monkeypatch.setattr(settings.langsmith, "api_key", None)

        warnings = validate_settings()

        assert any("TAVILY_API_KEY" in w for w in warnings)
        assert any("LANGCHAIN_API_KEY" in w for w in warnings)

    def test_invalid_config_value_fails_validation(self):
        """test-plan §8: invalid config values fail at construction, not silently coerce."""
        with pytest.raises(ValidationError):
            RetrySettings(max_attempts="not-a-number")

    def test_invalid_llm_timeout_fails_validation(self):
        with pytest.raises(ValidationError):
            LLMSettings(request_timeout_seconds=0)
