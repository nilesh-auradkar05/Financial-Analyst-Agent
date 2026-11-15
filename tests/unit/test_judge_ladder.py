"""Evaluation judge ladder: docs/test-plan.md §16 "Judge ladder" (ADR-0009 amendment 2026-10-17,
reordered 2026-11-14, CM-6c): Foundry judge deployment first, Bedrock Luna (bedrock-mantle) fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pytest
from pydantic import BaseModel

from evaluation.judge_models_interface import (
    JudgeModelConfig,
    JudgeProvider,
    create_deepeval_judge_model,
    create_langchain_judge_llm,
    judge_config_from_env,
)
from tests.unit.fake_openai import FakeOpenAIServer

MANTLE_HOST = "bedrock-mantle.us-east-1.api.aws"
AZURE_HOST = "alpha-test.openai.azure.com"
LUNA = "openai.gpt-5.6-luna"
DEPLOYMENT = "luna-judge"
AZURE_KEY = "dummy-azure-secret"
BEDROCK_KEY = "dummy-bedrock-secret"


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    server: FakeOpenAIServer,
    *,
    bedrock: bool = True,
    azure: bool = True,
) -> JudgeModelConfig:
    monkeypatch.setattr("langchain_openai.ChatOpenAI", server.chat_openai)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("DEEPEVAL_MODEL", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", BEDROCK_KEY if bedrock else "")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", f"https://{AZURE_HOST}")
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY if azure else "")
    monkeypatch.setenv("AZURE_FOUNDRY_JUDGE_MODEL", DEPLOYMENT if azure else "")
    return judge_config_from_env(
        provider="luna", model=None, temperature=0.0, top_p=None, max_tokens=None
    )


def _judge(
    monkeypatch: pytest.MonkeyPatch,
    server: FakeOpenAIServer,
    *,
    bedrock: bool = True,
    azure: bool = True,
) -> Any:
    return create_langchain_judge_llm(_configure(monkeypatch, server, bedrock=bedrock, azure=azure))


def _from_env(model: str | None) -> JudgeModelConfig:
    return judge_config_from_env(
        provider="luna", model=model, temperature=0.0, top_p=None, max_tokens=None
    )


def test_foundry_judges_first_with_the_azure_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row clauses: Foundry first via /openai/v1/; no temperature; host identifiable from model name."""
    server = FakeOpenAIServer()

    reply = _judge(monkeypatch, server).invoke("check")

    assert reply.content == f"{DEPLOYMENT}: check"
    assert reply.response_metadata["model_name"] == DEPLOYMENT  # the answering host is recorded
    [request] = server.requests  # a Foundry success never reaches bedrock-mantle
    assert str(request.url) == f"https://{AZURE_HOST}/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {AZURE_KEY}"
    assert server.bodies()[0]["model"] == DEPLOYMENT
    assert "temperature" not in server.bodies()[0]


@pytest.mark.parametrize("status", [500, 429, 401])
def test_foundry_failure_reaches_luna_on_bedrock_mantle(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Row clauses: a Foundry failure reaches Luna via bedrock-mantle with the Bedrock key; no temperature."""
    server = FakeOpenAIServer(failures={AZURE_HOST: status})

    reply = _judge(monkeypatch, server).invoke("check")

    assert reply.content == f"{LUNA}: check"
    assert reply.response_metadata["model_name"] == LUNA
    assert [r.url.host for r in server.requests] == [AZURE_HOST, MANTLE_HOST]
    assert str(server.requests[1].url) == f"https://{MANTLE_HOST}/openai/v1/chat/completions"
    assert server.requests[1].headers["authorization"] == f"Bearer {BEDROCK_KEY}"
    assert all("temperature" not in body for body in server.bodies())


def test_unconfigured_foundry_tier_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row clause: an unconfigured tier is skipped (Foundry), and the fallback answers alone."""
    server = FakeOpenAIServer()

    reply = _judge(monkeypatch, server, azure=False).invoke("check")

    assert reply.content == f"{LUNA}: check"
    assert [r.url.host for r in server.requests] == [MANTLE_HOST]


def test_unconfigured_bedrock_tier_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row clause: an unconfigured tier is skipped (Bedrock): a Foundry failure has nowhere to fall."""
    server = FakeOpenAIServer(failures={AZURE_HOST: 500})

    with pytest.raises(openai.InternalServerError):
        _judge(monkeypatch, server, bedrock=False).invoke("check")

    assert [r.url.host for r in server.requests] == [AZURE_HOST]


def test_no_configured_tier_is_refused_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row clause: no tier configured -> refused with a message naming the settings, no secret value."""
    with pytest.raises(ValueError) as caught:
        _judge(monkeypatch, FakeOpenAIServer(), bedrock=False, azure=False)

    message = str(caught.value)
    assert "AWS_BEARER_TOKEN_BEDROCK" in message
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "dummy" not in message


def test_deepeval_judge_gets_structured_verdicts_from_foundry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row clause: structured verdicts through the DeepEval adapter, Foundry answering first."""

    class Verdict(BaseModel):
        verdict: str

    server = FakeOpenAIServer()
    judge = create_deepeval_judge_model(_configure(monkeypatch, server))
    assert judge is not None and not isinstance(judge, str)

    result = judge.generate("Is the claim supported?", schema=Verdict)

    assert result == Verdict(verdict=DEPLOYMENT)
    assert [r.url.host for r in server.requests] == [AZURE_HOST]
    assert server.bodies()[0]["response_format"]["type"] == "json_schema"


def test_deepeval_judge_falls_back_to_bedrock_for_structured_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row clause: structured verdicts through the DeepEval adapter survive the fallback."""

    class Verdict(BaseModel):
        verdict: str

    server = FakeOpenAIServer(failures={AZURE_HOST: 500})
    judge = create_deepeval_judge_model(_configure(monkeypatch, server))
    assert judge is not None and not isinstance(judge, str)

    result = judge.generate("Is the claim supported?", schema=Verdict)

    assert result == Verdict(verdict=LUNA)
    assert [r.url.host for r in server.requests] == [AZURE_HOST, MANTLE_HOST]
    assert server.bodies()[1]["response_format"]["type"] == "json_schema"


def test_luna_config_is_the_foundry_deployment_when_foundry_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row clause: for provider luna the configured model (the eval output's judge_model) is the deployment."""
    config = _configure(monkeypatch, FakeOpenAIServer())

    assert config.provider == JudgeProvider.LUNA
    assert config.model == DEPLOYMENT


def test_luna_config_is_the_bedrock_id_when_foundry_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row clause: the Bedrock id is recorded when Foundry is unconfigured."""
    config = _configure(monkeypatch, FakeOpenAIServer(), azure=False)

    assert config.provider == JudgeProvider.LUNA
    assert config.model == LUNA


@pytest.mark.parametrize(
    "blank", ["AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY", "AZURE_FOUNDRY_JUDGE_MODEL"]
)
def test_partially_configured_foundry_counts_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """Row clause: Foundry is configured only when endpoint, key and judge deployment are all set."""
    _configure(monkeypatch, FakeOpenAIServer())
    monkeypatch.setenv(blank, "")

    assert _from_env(None).model == LUNA


@pytest.mark.parametrize("foundry", [True, False])
def test_judge_model_equal_to_the_settings_value_is_accepted(
    monkeypatch: pytest.MonkeyPatch, foundry: bool
) -> None:
    """Row clause: --judge-model is not required; the settings-derived value itself is accepted."""
    expected = DEPLOYMENT if foundry else LUNA
    _configure(monkeypatch, FakeOpenAIServer(), azure=foundry)

    assert _from_env(expected).model == expected
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", expected)
    assert _from_env(None).model == expected


@pytest.mark.parametrize(
    ("foundry", "other"),
    [(True, "gpt-4o"), (True, LUNA), (False, "gpt-4o"), (False, DEPLOYMENT)],
)
def test_a_different_judge_model_argument_is_rejected(
    monkeypatch: pytest.MonkeyPatch, foundry: bool, other: str
) -> None:
    """Row clause: a different --judge-model is rejected, naming the setting and no secret value."""
    _configure(monkeypatch, FakeOpenAIServer(), azure=foundry)

    with pytest.raises(ValueError) as caught:
        _from_env(other)

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "dummy" not in message


def test_a_different_judge_model_environment_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row clause: a different RAG_EVAL_JUDGE_MODEL value is rejected."""
    _configure(monkeypatch, FakeOpenAIServer())
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o")

    with pytest.raises(ValueError) as caught:
        _from_env(None)

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "dummy" not in message


def test_cli_does_not_require_judge_model_for_luna(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Row clause: --judge-model is not required for luna. Argument parsing must succeed, so the run
    fails later, on the missing samples file, instead of exiting with a usage error."""
    from evaluation.run_rag_quality_eval import main

    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("DEEPEVAL_MODEL", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rag_quality_eval",
            str(tmp_path / "missing-samples.json"),
            "--judge-provider",
            "luna",
        ],
    )

    try:
        main()
    except SystemExit as exit_:
        pytest.fail(
            f"argument parsing exited with status {exit_.code}; --judge-model must be optional"
        )
    except Exception:  # noqa: BLE001 - any post-parse failure (missing samples file) is the expected one
        return
    pytest.fail("main() returned although the samples file does not exist")
