"""Offline provider behavior from docs/test-plan.md §16."""

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr

from app.config import LLMSettings, Settings
from app.llm import provider


class EchoModel(BaseChatModel):
    provider_name: str
    delay: float = 0

    @property
    def _llm_type(self) -> str:
        return self.provider_name

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = f"{self.provider_name}: {messages[-1].content}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._generate(messages)


class TraceEvents(BaseCallbackHandler):
    def __init__(self) -> None:
        self.children: list[object] = []

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self.children.append(kwargs.get("parent_run_id"))


def _settings() -> Settings:
    llm = LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    llm.aws_region = "us-east-1"
    llm.model = "anthropic.claude-sonnet-5-5"
    llm.aws_bearer_token_bedrock = None
    llm.anthropic_api_key = None
    llm.openai_api_key = None
    llm.request_timeout_seconds = 0.05
    return Settings(llm=llm, _env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


def test_bedrock_model_variable_and_bearer_alias_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BEDROCK_MODEL", "canonical-model")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "canonical-dummy")
    monkeypatch.setenv("AWS_BEARER_TOKEN", "legacy-dummy")
    config = LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    assert config.model == "canonical-model"
    assert config.aws_bearer_token_bedrock == SecretStr("canonical-dummy")


@pytest.mark.parametrize("name", ["LLM_MODEL", "CLAUDE_LLM_MODEL", "OPENAI_LLM_MODEL"])
def test_personal_model_variables_never_select_bedrock_model(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.delenv("AWS_BEDROCK_MODEL", raising=False)
    monkeypatch.setenv(name, "personal-model")
    assert LLMSettings(_env_file=None).model == "global.anthropic.claude-sonnet-4-6"  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic_key, expected", [("dummy", "claude-personal: check"), (None, "gpt-personal: check")])
async def test_personal_model_variables_reach_their_provider_in_order(
    monkeypatch: pytest.MonkeyPatch, anthropic_key: str | None, expected: str,
) -> None:
    monkeypatch.setenv("CLAUDE_LLM_MODEL", "claude-personal")
    monkeypatch.setenv("OPENAI_LLM_MODEL", "gpt-personal")
    llm = LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    llm.aws_region = "us-east-1"
    llm.anthropic_api_key = SecretStr(anthropic_key) if anthropic_key else None
    llm.openai_api_key = SecretStr("dummy")
    settings = Settings(llm=llm, _env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]

    def broken_bedrock(_: Settings) -> EchoModel:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(provider, "_bedrock_model", broken_bedrock)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", lambda **kw: EchoModel(provider_name=kw["model"]))
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: EchoModel(provider_name=kw["model"]))
    assert provider.fallback_count(settings) == (3 if anthropic_key else 2)
    assert (await provider.get_llm(settings).ainvoke("check")).content == expected


def test_default_model_is_invocable_inference_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_BEDROCK_MODEL", "CLAUDE_LLM_MODEL", "OPENAI_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    config = LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    assert config.model == "global.anthropic.claude-sonnet-4-6"


def test_bedrock_sonnet_5_preserves_selection_without_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    settings.llm.aws_bearer_token_bedrock = SecretStr("dummy")
    captured: dict[str, Any] = {}

    def build(**kwargs: Any) -> EchoModel:
        captured.update(kwargs)
        return EchoModel(provider_name="bedrock")

    monkeypatch.setattr("langchain_aws.ChatBedrockConverse", build)
    assert provider.get_llm(settings).invoke("check").content == "bedrock: check"
    assert captured["model"] == settings.llm.model
    assert captured["region_name"] == "us-east-1"
    assert captured["bedrock_api_key"] == SecretStr("dummy")
    assert "temperature" not in captured


@pytest.mark.asyncio
async def test_primary_success_does_not_construct_personal_models(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    settings.llm.anthropic_api_key = SecretStr("dummy")
    settings.llm.openai_api_key = SecretStr("dummy")
    monkeypatch.setattr(provider, "_bedrock_model", lambda _: EchoModel(provider_name="bedrock"))

    def unexpected(**kwargs: Any) -> EchoModel:
        raise AssertionError("unused fallback constructed")

    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", unexpected)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", unexpected)
    response = await provider.get_llm(settings).ainvoke("check")
    assert response.content == "bedrock: check"


@pytest.mark.asyncio
async def test_construction_and_timeout_failures_reach_next_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings.llm.anthropic_api_key = SecretStr("dummy")
    settings.llm.openai_api_key = SecretStr("dummy")

    def broken_bedrock(_: Settings) -> EchoModel:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(provider, "_bedrock_model", broken_bedrock)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", lambda **_: EchoModel(provider_name="anthropic", delay=0.2))
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **_: EchoModel(provider_name="openai"))
    events = TraceEvents()
    response = await provider.get_llm(settings).ainvoke("check", config={"callbacks": [events]})
    assert response.content == "openai: check"
    assert len(events.children) == 2
    assert all(parent is not None for parent in events.children)


@pytest.mark.asyncio
async def test_all_failed_attempts_raise_without_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    settings.llm.anthropic_api_key = SecretStr("dummy")

    def broken(_: Settings) -> EchoModel:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(provider, "_bedrock_model", broken)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", lambda **_: EchoModel(provider_name="anthropic", delay=0.2))
    with pytest.raises(RuntimeError, match="unavailable"):
        await provider.get_llm(settings).ainvoke("check")


def test_unconfigured_fallbacks_are_skipped() -> None:
    settings = _settings()
    assert provider.fallback_count(settings) == 1
    settings.llm.anthropic_api_key = SecretStr(" ")
    assert provider.fallback_count(settings) == 1


@pytest.mark.asyncio
async def test_health_accepts_configured_personal_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    async def embedding_available(*, model: str | None = None) -> bool:
        return True

    async def langsmith_status() -> dict[str, bool]:
        return {"connected": False}

    class Store:
        count = 0

    monkeypatch.setattr(main.settings.llm, "aws_region", None)
    monkeypatch.setattr(main.settings.llm, "anthropic_api_key", SecretStr("dummy"))
    monkeypatch.setattr(main, "check_ollama_health", embedding_available)
    monkeypatch.setattr(main, "check_langsmith_connection", langsmith_status)
    monkeypatch.setattr(main, "get_vector_store", lambda: Store())

    response = await main.health()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_model_is_built_once_and_a_failed_build_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace: docs/test-plan.md §10, chat model built once per process (S2-T00c)."""
    settings = _settings()
    settings.llm.request_timeout_seconds = 1
    built: list[EchoModel] = []

    def bedrock(_: Settings) -> EchoModel:
        built.append(EchoModel(provider_name=f"bedrock-build-{len(built)}"))
        if len(built) == 1:
            raise RuntimeError("cold-start failure")
        return built[-1]

    monkeypatch.setattr(provider, "_bedrock_model", bedrock)

    with pytest.raises(RuntimeError):
        await provider.get_llm(settings).ainvoke("check")
    first = await provider.get_llm(settings).ainvoke("check")
    second = await provider.get_llm(settings).ainvoke("check")

    assert first.content == second.content == "bedrock-build-1: check"
