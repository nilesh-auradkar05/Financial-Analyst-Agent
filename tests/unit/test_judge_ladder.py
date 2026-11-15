"""Evaluation judge ladder: docs/test-plan.md §16 "Judge ladder (Claude Opus 4.6, 2026-11-15,
JUDGE-OPUS; supersedes the Luna row)" (ADR-0009 amendment 2026-11-15): the Foundry judge deployment
first through the Anthropic Messages route, Bedrock Opus 4.6 through the Converse runtime as the
fallback.

Both providers are loopback fakes that compute each reply from the request they receive, so the
production clients build and send real requests: FakeFoundry for Foundry and FakeBedrockConverse
(reached through `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`) for Bedrock.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from evaluation.judge_models_interface import (
    JudgeModelConfig,
    create_deepeval_judge_model,
    create_langchain_judge_llm,
    judge_config_from_env,
)
from tests.embedding_arms import isolate_environment
from tests.unit.fake_bedrock_converse import FakeBedrockConverse
from tests.unit.fake_foundry import JUDGE_SERVED_MODEL, MESSAGES_PATH, FakeFoundry

DEPLOYMENT = "opus-4-6-judge"
BEDROCK_ID = "global.anthropic.claude-opus-4-6-v1"
AZURE_KEY = "dummy-azure-secret"
BEDROCK_KEY = "dummy-bedrock-secret"
SECRETS = (AZURE_KEY, BEDROCK_KEY)


class Verdict(BaseModel):
    verdict: str
    reason: str


@dataclass
class Ladder:
    """The two loopback providers and the environment that points the judge at them."""

    monkeypatch: pytest.MonkeyPatch
    workdir: Path
    foundry: FakeFoundry
    bedrock: FakeBedrockConverse

    def configure(self, *, azure: bool = True, bedrock: bool = True) -> None:
        isolate_environment(self.monkeypatch, self.workdir)
        for name in ("RAG_EVAL_JUDGE_MODEL", "DEEPEVAL_MODEL"):
            self.monkeypatch.delenv(name, raising=False)
        env = {
            "AZURE_OPENAI_ENDPOINT": self.foundry.url if azure else "",
            "AZURE_API_KEY": AZURE_KEY if azure else "",
            "AZURE_FOUNDRY_JUDGE_MODEL": DEPLOYMENT if azure else "",
            "AWS_BEARER_TOKEN_BEDROCK": BEDROCK_KEY if bedrock else "",
            "AWS_REGION": "us-east-1",
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": self.bedrock.url,
        }
        for name, value in env.items():
            self.monkeypatch.setenv(name, value)

    def config(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        provider: str = "opus",
    ) -> JudgeModelConfig:
        return judge_config_from_env(
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
        )

    def judge(
        self,
        *,
        azure: bool = True,
        bedrock: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        self.configure(azure=azure, bedrock=bedrock)
        return self.build(temperature=temperature, max_tokens=max_tokens)

    def build(self, **options: Any) -> Any:
        """The judge for the current environment (the environment is configured by the caller)."""
        judge = create_langchain_judge_llm(self.config(**options))
        assert judge is not None
        return judge

    def deepeval(self) -> Any:
        self.configure()
        judge = create_deepeval_judge_model(self.config())
        assert judge is not None and not isinstance(judge, str)
        return judge


@pytest.fixture
def ladder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Ladder]:
    foundry = FakeFoundry()
    foundry.judge_replies = True
    foundry.start()
    bedrock = FakeBedrockConverse().start()
    yield Ladder(monkeypatch, tmp_path, foundry, bedrock)
    foundry.stop()
    bedrock.stop()


@pytest.mark.parametrize("temperature", [0.0, 0.3])
def test_foundry_judges_first_over_anthropic_messages(ladder: Ladder, temperature: float) -> None:
    """Row clauses: Foundry first via {AZURE_OPENAI_ENDPOINT}/anthropic/v1/messages, AZURE_API_KEY in
    x-api-key and no bearer header; the configured temperature, no top_p, no thinking, explicit
    max_tokens; Bedrock is not reached."""
    reply = ladder.judge(temperature=temperature).invoke("check")

    assert reply.content == f"{DEPLOYMENT}: check"
    [request] = ladder.foundry.requests
    assert request.path == MESSAGES_PATH
    assert request.credential_header == "x-api-key"
    assert request.credential == AZURE_KEY
    assert not request.bearer
    assert request.body["model"] == DEPLOYMENT
    assert request.body["temperature"] == temperature
    assert "top_p" not in request.body
    assert "thinking" not in request.body
    assert isinstance(request.body["max_tokens"], int) and request.body["max_tokens"] > 0
    assert ladder.bedrock.requests == []


@pytest.mark.parametrize("temperature", [0.0, 0.3])
@pytest.mark.parametrize("status", [500, 429, 401])
def test_foundry_failure_reaches_bedrock_opus_on_the_converse_runtime(
    ladder: Ladder, status: int, temperature: float
) -> None:
    """Row clauses: a Foundry failure reaches Bedrock Opus 4.6 through the Converse runtime with the
    Bedrock API key; the configured temperature, no top_p, no thinking, explicit max_tokens; the
    Foundry tier makes one attempt (no client retries)."""
    ladder.foundry.status = status

    reply = ladder.judge(temperature=temperature).invoke("check")

    assert reply.content == f"{BEDROCK_ID}: check"
    assert len(ladder.foundry.requests) == 1
    [request] = ladder.bedrock.requests
    assert request.path == f"/model/{BEDROCK_ID}/converse"
    assert request.bearer == BEDROCK_KEY
    inference = request.body["inferenceConfig"]
    assert inference["temperature"] == temperature
    assert "topP" not in inference
    assert isinstance(inference["maxTokens"], int) and inference["maxTokens"] > 0
    assert "thinking" not in (request.body.get("additionalModelRequestFields") or {})


@pytest.mark.parametrize("max_tokens", [None, 777])
def test_both_tiers_send_an_explicit_max_tokens(ladder: Ladder, max_tokens: int | None) -> None:
    """Row clause: an explicit max_tokens on both tiers (the configured one when there is one)."""
    judge = ladder.judge(max_tokens=max_tokens)
    judge.invoke("first")
    ladder.foundry.status = 500
    judge.invoke("second")

    foundry_request = ladder.foundry.requests[0]
    [bedrock_request] = ladder.bedrock.requests
    sent = [
        foundry_request.body["max_tokens"],
        bedrock_request.body["inferenceConfig"]["maxTokens"],
    ]
    assert all(isinstance(value, int) and value > 0 for value in sent)
    if max_tokens is not None:
        assert sent == [max_tokens, max_tokens]
    else:
        assert sent[0] == sent[1]


def test_every_tier_fails_after_one_attempt_each(ladder: Ladder) -> None:
    """Row clause: no client retries on either tier; a provider failure surfaces once both failed."""
    ladder.foundry.status = 500
    ladder.bedrock.status = 500

    with pytest.raises((anthropic.APIStatusError, ClientError)):
        ladder.judge().invoke("check")

    assert len(ladder.foundry.requests) == 1
    assert len(ladder.bedrock.requests) == 1


def test_unconfigured_foundry_tier_is_skipped(ladder: Ladder) -> None:
    """Row clause: an unconfigured tier is skipped (Foundry); Bedrock answers alone."""
    reply = ladder.judge(azure=False).invoke("check")

    assert reply.content == f"{BEDROCK_ID}: check"
    assert ladder.foundry.requests == []
    assert len(ladder.bedrock.requests) == 1


@pytest.mark.parametrize(
    "blank", ["AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY", "AZURE_FOUNDRY_JUDGE_MODEL"]
)
def test_partially_configured_foundry_is_skipped(ladder: Ladder, blank: str) -> None:
    """Row clause: an unconfigured tier is skipped; Foundry needs endpoint, key and judge deployment."""
    ladder.configure()
    ladder.monkeypatch.setenv(blank, "")

    reply = ladder.build().invoke("check")

    assert reply.content == f"{BEDROCK_ID}: check"
    assert ladder.foundry.requests == []


@pytest.mark.parametrize("unset", ["AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"])
def test_unconfigured_bedrock_tier_is_skipped(ladder: Ladder, unset: str) -> None:
    """Row clause: an unconfigured tier is skipped (Bedrock needs the API key and AWS_REGION): a
    Foundry failure has nowhere to fall and its own error surfaces."""
    ladder.foundry.status = 500
    ladder.configure()
    ladder.monkeypatch.delenv(unset)
    judge = ladder.build()

    with pytest.raises(anthropic.APIStatusError):
        judge.invoke("check")

    assert len(ladder.foundry.requests) == 1
    assert ladder.bedrock.requests == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"AZURE_OPENAI_ENDPOINT": "", "AZURE_API_KEY": "", "AZURE_FOUNDRY_JUDGE_MODEL": ""},
        {"AZURE_FOUNDRY_JUDGE_MODEL": ""},
        {"AZURE_API_KEY": ""},
    ],
    ids=["nothing-set", "azure-key-without-deployment", "deployment-without-azure-key"],
)
def test_no_configured_tier_is_refused_without_secrets(
    ladder: Ladder, overrides: dict[str, str]
) -> None:
    """Row clause: with no tier configured the judge is refused with a message naming the settings
    and no secret value (even when a secret is set but its tier is incomplete)."""
    ladder.configure(bedrock=False)
    for name, value in overrides.items():
        ladder.monkeypatch.setenv(name, value)

    with pytest.raises(ValueError) as caught:
        create_langchain_judge_llm(ladder.config())

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "AWS_BEARER_TOKEN_BEDROCK" in message
    assert not any(secret in message for secret in SECRETS)


def test_a_bedrock_key_without_a_region_is_no_configured_tier(ladder: Ladder) -> None:
    """Row clause: with no tier configured the judge is refused, no secret value in the message
    (the Bedrock key is set but AWS_REGION is not)."""
    ladder.configure(azure=False)
    ladder.monkeypatch.delenv("AWS_REGION")

    with pytest.raises(ValueError) as caught:
        create_langchain_judge_llm(ladder.config())

    message = str(caught.value)
    assert "AWS_BEARER_TOKEN_BEDROCK" in message
    assert not any(secret in message for secret in SECRETS)


def test_answering_host_is_identifiable_from_the_reply_model_name(ladder: Ladder) -> None:
    """Row clause: the answering host is identifiable from the reply's model name: the id Foundry
    serves when it answers, the Bedrock id after a fallback."""
    judge = ladder.judge()

    from_foundry = judge.invoke("check").response_metadata["model_name"]
    ladder.foundry.status = 500
    from_bedrock = judge.invoke("check").response_metadata["model_name"]

    assert from_foundry == JUDGE_SERVED_MODEL
    assert from_bedrock == BEDROCK_ID
    assert from_foundry != from_bedrock


def test_deepeval_judge_gets_structured_verdicts_from_foundry(ladder: Ladder) -> None:
    """Row clause: structured verdicts through the DeepEval adapter, Foundry answering first."""
    result = ladder.deepeval().generate("Is the claim supported?", schema=Verdict)

    assert result == Verdict(verdict=DEPLOYMENT, reason=DEPLOYMENT)
    [request] = ladder.foundry.requests
    [tool] = request.body["tools"]
    assert set(tool["input_schema"]["properties"]) == {"verdict", "reason"}
    assert ladder.bedrock.requests == []


def test_deepeval_judge_falls_back_to_bedrock_for_structured_verdicts(ladder: Ladder) -> None:
    """Row clause: structured verdicts through the DeepEval adapter survive the fallback."""
    ladder.foundry.status = 500

    result = ladder.deepeval().generate("Is the claim supported?", schema=Verdict)

    assert result == Verdict(verdict=BEDROCK_ID, reason=BEDROCK_ID)
    assert len(ladder.foundry.requests) == 1
    [request] = ladder.bedrock.requests
    [tool] = request.body["toolConfig"]["tools"]
    assert set(tool["toolSpec"]["inputSchema"]["json"]["properties"]) == {"verdict", "reason"}


def test_opus_config_is_the_foundry_deployment_when_foundry_is_configured(ladder: Ladder) -> None:
    """Row clause: for provider opus the configured model (the eval output's judge_model) is the
    Foundry deployment."""
    ladder.configure()

    config = ladder.config()

    assert config.provider == "opus"
    assert config.model == DEPLOYMENT


def test_opus_config_is_the_bedrock_id_when_foundry_is_unconfigured(ladder: Ladder) -> None:
    """Row clause: the Bedrock id is the configured model when Foundry is unconfigured."""
    ladder.configure(azure=False)

    config = ladder.config()

    assert config.provider == "opus"
    assert config.model == BEDROCK_ID


@pytest.mark.parametrize(
    "blank", ["AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY", "AZURE_FOUNDRY_JUDGE_MODEL"]
)
def test_partially_configured_foundry_records_the_bedrock_id(ladder: Ladder, blank: str) -> None:
    """Row clause: Foundry is configured only when endpoint, key and judge deployment are all set."""
    ladder.configure()
    ladder.monkeypatch.setenv(blank, "")

    assert ladder.config().model == BEDROCK_ID


@pytest.mark.parametrize("foundry", [True, False])
def test_judge_model_equal_to_the_settings_value_is_accepted(ladder: Ladder, foundry: bool) -> None:
    """Row clause: --judge-model is not required; the settings-derived value itself is accepted,
    as an argument and in RAG_EVAL_JUDGE_MODEL."""
    expected = DEPLOYMENT if foundry else BEDROCK_ID
    ladder.configure(azure=foundry)

    assert ladder.config(model=expected).model == expected
    ladder.monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", expected)
    assert ladder.config().model == expected


@pytest.mark.parametrize(
    ("foundry", "other"),
    [
        (True, "gpt-4o"),
        (True, BEDROCK_ID),
        (False, "gpt-4o"),
        (False, DEPLOYMENT),
        (True, "openai.gpt-5.6-luna"),
    ],
)
def test_a_different_judge_model_argument_is_rejected(
    ladder: Ladder, foundry: bool, other: str
) -> None:
    """Row clause: a different --judge-model is rejected, naming the settings and no secret value."""
    ladder.configure(azure=foundry)

    with pytest.raises(ValueError) as caught:
        ladder.config(model=other)

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "DEEPEVAL_MODEL" in message
    assert "dummy" not in message


def test_a_different_judge_model_environment_value_is_rejected(ladder: Ladder) -> None:
    """Row clause: a different RAG_EVAL_JUDGE_MODEL value is rejected, naming the settings and no
    secret value."""
    ladder.configure()
    ladder.monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o")

    with pytest.raises(ValueError) as caught:
        ladder.config()

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "DEEPEVAL_MODEL" in message
    assert "dummy" not in message


def test_provider_luna_is_rejected(ladder: Ladder) -> None:
    """Row clause: provider luna is rejected."""
    ladder.configure()

    with pytest.raises(ValueError):
        ladder.config(provider="luna")


def test_cli_does_not_require_judge_model_for_opus(
    ladder: Ladder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Row clause: --judge-model is not required for opus. Argument parsing must succeed, so the run
    fails later, on the missing samples file, instead of exiting with a usage error."""
    from evaluation.run_rag_quality_eval import main

    ladder.configure()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rag_quality_eval",
            str(tmp_path / "missing-samples.json"),
            "--judge-provider",
            "opus",
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


def test_cli_rejects_a_different_deepeval_model_without_calling_a_provider(
    ladder: Ladder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Row clause: a different DEEPEVAL_MODEL value is rejected (the CLI feeds it in as the judge
    model), naming the settings and no secret value, before any provider is called."""
    from evaluation.run_rag_quality_eval import main

    ladder.configure()
    monkeypatch.setenv("DEEPEVAL_MODEL", "gpt-4o")
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "ticker": "AAPL",
                    "input": "What drove revenue?",
                    "actual_output": "Services revenue grew.",
                    "expected_output": "Services revenue grew.",
                    "retrieval_context": ["Services revenue grew 14%."],
                    "context": ["Services revenue grew 14%."],
                }
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv", ["run_rag_quality_eval", str(samples), "--judge-provider", "opus"]
    )

    with pytest.raises(ValueError) as caught:
        main()

    message = str(caught.value)
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "DEEPEVAL_MODEL" in message
    assert not any(secret in message for secret in SECRETS)
    assert ladder.foundry.requests == []
    assert ladder.bedrock.requests == []
