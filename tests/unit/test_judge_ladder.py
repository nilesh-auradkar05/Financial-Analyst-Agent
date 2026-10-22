"""Evaluation judge ladder: docs/test-plan.md §16 "Judge ladder" (ADR-0009 amendment 2026-10-17)."""

from __future__ import annotations

from typing import Any

import openai
import pytest

from evaluation.judge_models_interface import (
    JudgeProvider,
    create_langchain_judge_llm,
    judge_config_from_env,
)
from tests.unit.fake_openai import FakeOpenAIServer

MANTLE_HOST = "bedrock-mantle.us-east-1.api.aws"
AZURE_HOST = "alpha-test.openai.azure.com"
LUNA = "openai.gpt-5.6-luna"


def _judge(
    monkeypatch: pytest.MonkeyPatch,
    server: FakeOpenAIServer,
    *,
    bedrock: bool = True,
    azure: bool = True,
) -> Any:
    monkeypatch.setattr("langchain_openai.ChatOpenAI", server.chat_openai)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy-bedrock" if bedrock else "")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", f"https://{AZURE_HOST}")
    monkeypatch.setenv("AZURE_API_KEY", "dummy-azure" if azure else "")
    monkeypatch.setenv("AZURE_FOUNDRY_JUDGE_MODEL", "luna-judge" if azure else "")
    config = judge_config_from_env(
        provider="luna", model=None, temperature=0.0, top_p=None, max_tokens=None
    )
    return create_langchain_judge_llm(config)


def test_luna_judge_defaults_to_the_bedrock_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    config = judge_config_from_env(
        provider="luna", model=None, temperature=0.0, top_p=None, max_tokens=None
    )
    assert config.provider == JudgeProvider.LUNA
    assert config.model == LUNA


def test_bedrock_mantle_judges_first_with_the_bedrock_key(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeOpenAIServer()

    reply = _judge(monkeypatch, server).invoke("check")

    assert reply.content == f"{LUNA}: check"
    assert reply.response_metadata["model_name"] == LUNA  # the answering host is recorded
    [request] = server.requests
    assert str(request.url) == f"https://{MANTLE_HOST}/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer dummy-bedrock"
    assert "temperature" not in server.bodies()[0]


def test_bedrock_failure_reaches_the_foundry_judge_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeOpenAIServer(failures={MANTLE_HOST: 403})

    reply = _judge(monkeypatch, server).invoke("check")

    assert reply.content == "luna-judge: check"
    assert reply.response_metadata["model_name"] == "luna-judge"
    assert [r.url.host for r in server.requests] == [MANTLE_HOST, AZURE_HOST]
    assert str(server.requests[1].url) == f"https://{AZURE_HOST}/openai/v1/chat/completions"
    assert server.requests[1].headers["authorization"] == "Bearer dummy-azure"


def test_unconfigured_bedrock_tier_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeOpenAIServer()

    reply = _judge(monkeypatch, server, bedrock=False).invoke("check")

    assert reply.content == "luna-judge: check"
    assert [r.url.host for r in server.requests] == [AZURE_HOST]


def test_unconfigured_azure_tier_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeOpenAIServer(failures={MANTLE_HOST: 500})

    with pytest.raises(openai.InternalServerError):
        _judge(monkeypatch, server, azure=False).invoke("check")

    assert [r.url.host for r in server.requests] == [MANTLE_HOST]


def test_no_configured_tier_is_refused_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError) as caught:
        _judge(monkeypatch, FakeOpenAIServer(), bedrock=False, azure=False)

    message = str(caught.value)
    assert "AWS_BEARER_TOKEN_BEDROCK" in message
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "dummy" not in message


def test_deepeval_judge_falls_back_to_foundry_for_structured_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The path a real eval takes: DeepEval adapter → structured output → fallback tier."""
    from pydantic import BaseModel

    from evaluation.judge_models_interface import create_deepeval_judge_model

    class Verdict(BaseModel):
        verdict: str

    server = FakeOpenAIServer(failures={MANTLE_HOST: 403})
    _judge(monkeypatch, server)  # configures both tiers and routes them to the fake
    config = judge_config_from_env(provider="luna", model=None, temperature=0.0, top_p=None, max_tokens=None)

    result = create_deepeval_judge_model(config).generate("Is the claim supported?", schema=Verdict)

    assert result == Verdict(verdict="luna-judge")
    assert [r.url.host for r in server.requests] == [MANTLE_HOST, AZURE_HOST]
    assert server.bodies()[1]["response_format"]["type"] == "json_schema"
