from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from evaluation.judge_models_interface import (
    JudgeProvider,
    create_ragas_judge_llm,
    judge_config_from_env,
)


def test_anthropic_defaults_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")

    config = judge_config_from_env(
        provider="langchain_anthropic",
        model=None,
        temperature=0.0,
        top_p=None,
        max_tokens=None,
    )

    assert config.provider == JudgeProvider.LANGCHAIN_ANTHROPIC
    assert config.model == "claude-sonnet-4-6"
    assert config.max_tokens == 4096


def test_ragas_default_uses_async_openai_client(monkeypatch):
    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
        ) -> None:
            self.api_key = api_key

    captured: dict[str, Any] = {}

    def fake_llm_factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "fake-llm"

    fake_openai = ModuleType("openai")
    setattr(fake_openai, "AsyncOpenAI", FakeAsyncOpenAI)

    fake_ragas = ModuleType("ragas")
    fake_ragas_llms = ModuleType("ragas.llms")
    setattr(fake_ragas_llms, "llm_factory", fake_llm_factory)

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.llms", fake_ragas_llms)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    config = judge_config_from_env(
        provider="deepeval_default",
        model="gpt-test",
        temperature=0.0,
        top_p=None,
        max_tokens=None,
    )

    assert create_ragas_judge_llm(config) == "fake-llm"
    assert captured["args"] == ("gpt-test",)
    assert captured["kwargs"]["provider"] == "openai"
    assert isinstance(captured["kwargs"]["client"], FakeAsyncOpenAI)
    assert captured["kwargs"]["client"].api_key == "openai-test-key"
