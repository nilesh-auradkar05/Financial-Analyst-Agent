from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from evaluation.judge_models_interface import (
    JudgeProvider,
    create_ragas_judge_embeddings,
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


def test_huggingface_embeddings_default_to_cached_cpu(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    fake_ragas = ModuleType("ragas")
    fake_ragas_embeddings = ModuleType("ragas.embeddings")
    setattr(fake_ragas_embeddings, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)

    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", fake_ragas_embeddings)
    monkeypatch.setenv("RAGAS_EMBEDDINGS_PROVIDER", "huggingface")
    monkeypatch.setenv("RAGAS_EMBEDDINGS_MODEL", "custom/model")
    monkeypatch.delenv("RAGAS_EMBEDDINGS_DEVICE", raising=False)
    monkeypatch.delenv("RAGAS_EMBEDDINGS_LOCAL_FILES_ONLY", raising=False)

    config = judge_config_from_env(
        provider="langchain_anthropic",
        model="claude-sonnet-4-6",
        temperature=0.0,
        top_p=None,
        max_tokens=None,
    )

    create_ragas_judge_embeddings(config)

    assert captured == {
        "model": "custom/model",
        "device": "cpu",
        "local_files_only": True,
    }


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
