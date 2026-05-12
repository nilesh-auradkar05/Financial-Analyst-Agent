from __future__ import annotations

from evaluation.judge_models_interface import JudgeProvider, judge_config_from_env


def test_chat_nvidia_defaults_from_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.delenv("RAG_EVAL_JUDGE_EXTRA_BODY", raising=False)

    config = judge_config_from_env(
        provider="chat_nvidia",
        model=None,
        temperature=0.0,
        top_p=None,
        max_tokens=None,
        extra_body_json=None,
    )

    assert config.provider == JudgeProvider.CHAT_NVIDIA
    assert config.model == "deepseek-ai/deepseek-v4-pro"
    assert config.top_p == 0.95
    assert config.max_tokens == 16384
    assert config.extra_body == {"chat_template_kwargs": {"thinking_mode": False}}


def test_anthropic_defaults_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")

    config = judge_config_from_env(
        provider="langchain_anthropic",
        model=None,
        temperature=0.0,
        top_p=None,
        max_tokens=None,
        extra_body_json=None,
    )

    assert config.provider == JudgeProvider.LANGCHAIN_ANTHROPIC
    assert config.model == "claude-haiku-4-5-20251001"
    assert config.max_tokens == 4096


def test_extra_body_json_override():
    config = judge_config_from_env(
        provider="chat_nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
        temperature=0.0,
        top_p=0.1,
        max_tokens=1024,
        extra_body_json='{"chat_template_kwargs": {"thinking": false}, "foo": "bar"}',
    )

    assert config.model == "nvidia/nemotron-3-super-120b-a12b"
    assert config.top_p == 0.1
    assert config.max_tokens == 1024
    assert config.extra_body["foo"] == "bar"
