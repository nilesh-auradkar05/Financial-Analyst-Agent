"""Tests for app.config — security defaults and validation (test-plan §8)."""

import pytest
from pydantic import ValidationError

from app.config import (
    LLMSettings,
    RetrySettings,
    Settings,
    settings,
    validate_settings,
)


class TestSecurityDefaults:
    def test_api_security_defaults_fail_closed(self, monkeypatch):
        # deepeval's pytest plugin autoloads .env into os.environ; isolate from it.
        monkeypatch.delenv("API_KEY", raising=False)
        custom = Settings(_env_file=None)
        assert custom.api_key is None
        assert custom.api_rate_limit == 10


class TestValidation:
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
