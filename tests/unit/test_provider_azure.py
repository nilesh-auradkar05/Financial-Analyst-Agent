"""Azure-first draft ladder: docs/test-plan.md §16 rows dated 2026-10-17 (ADR-0009 amendment)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr, ValidationError

from app.config import LLMSettings, Settings
from app.llm import provider
from tests.unit.fake_openai import FakeOpenAIServer
from tests.unit.test_provider_fallback import EchoModel, _settings

AZURE_HOST = "alpha-test.openai.azure.com"
DEPLOYMENT = "ds-v4-flash-draft"


def _azure_settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    values = {
        "AZURE_OPENAI_ENDPOINT": f"https://{AZURE_HOST}",
        "AZURE_API_KEY": "dummy-azure",
        "AZURE_FOUNDRY_MODEL": DEPLOYMENT,
        **overrides,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = _settings()
    settings.llm.request_timeout_seconds = 1
    return settings


def _serve(monkeypatch: pytest.MonkeyPatch, server: FakeOpenAIServer) -> None:
    monkeypatch.setattr("langchain_openai.ChatOpenAI", server.chat_openai)
    monkeypatch.setattr(provider, "_bedrock_model", lambda _: EchoModel(provider_name="bedrock"))


@pytest.mark.asyncio
async def test_azure_is_the_first_attempt_and_bedrock_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _azure_settings(monkeypatch)
    server = FakeOpenAIServer()
    monkeypatch.setattr("langchain_openai.ChatOpenAI", server.chat_openai)

    def unexpected(_: Settings) -> EchoModel:
        raise AssertionError("Bedrock built although Azure answered")

    monkeypatch.setattr(provider, "_bedrock_model", unexpected)

    response = await provider.get_llm(settings).ainvoke("check")

    assert response.content == f"{DEPLOYMENT}: check"
    [request] = server.requests
    assert str(request.url) == f"https://{AZURE_HOST}/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer dummy-azure"
    assert server.bodies()[0]["model"] == DEPLOYMENT
    assert server.bodies()[0]["max_completion_tokens"] == settings.llm.max_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY", "AZURE_FOUNDRY_MODEL"])
async def test_incomplete_azure_configuration_is_skipped(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    settings = _azure_settings(monkeypatch, **{blank: " "})
    server = FakeOpenAIServer()
    _serve(monkeypatch, server)

    response = await provider.get_llm(settings).ainvoke("check")

    assert response.content == "bedrock: check"
    assert server.requests == []
    assert provider.fallback_count(settings) == 1
    assert provider.model_metadata(settings)["provider"] == "bedrock"


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["/openai/v1", "/openai/v1/", "/"])
async def test_endpoint_suffix_is_normalized(monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    settings = _azure_settings(monkeypatch, AZURE_OPENAI_ENDPOINT=f"https://{AZURE_HOST}{suffix}")
    server = FakeOpenAIServer()
    _serve(monkeypatch, server)

    await provider.get_llm(settings).ainvoke("check")

    assert str(server.requests[0].url) == f"https://{AZURE_HOST}/openai/v1/chat/completions"


def test_project_endpoint_is_rejected_when_settings_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", f"https://{AZURE_HOST}/api/projects/alpha")
    monkeypatch.setenv("AZURE_API_KEY", "secret-azure-key")

    with pytest.raises(ValidationError) as caught:
        LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]

    assert "AZURE_OPENAI_ENDPOINT" in str(caught.value)
    assert "secret-azure-key" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [429, 500, 401, "timeout"])
async def test_azure_failure_reaches_bedrock_in_the_same_run(
    monkeypatch: pytest.MonkeyPatch,
    failure: int | str,
) -> None:
    settings = _azure_settings(monkeypatch)
    settings.llm.request_timeout_seconds = 1  # margin for the thread hop that builds the client
    server = (
        FakeOpenAIServer(delay=5)
        if failure == "timeout"
        else FakeOpenAIServer(failures={AZURE_HOST: int(failure)})
    )
    _serve(monkeypatch, server)

    response = await provider.get_llm(settings).ainvoke("check")

    assert response.content == "bedrock: check"
    assert len(server.requests) == 1  # one Azure attempt, never retried inside the attempt


class AttemptNames(BaseCallbackHandler):
    def __init__(self) -> None:
        self.chains: list[str] = []
        self.models: list[str] = []

    def on_chain_start(self, serialized: dict, inputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.chains.append(kwargs.get("name") or "")

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self.models.append((kwargs.get("metadata") or {}).get("ls_model_name", ""))


@pytest.mark.asyncio
async def test_azure_attempt_is_traced_with_provider_and_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _azure_settings(monkeypatch)
    _serve(monkeypatch, FakeOpenAIServer())
    names = AttemptNames()

    await provider.get_llm(settings).ainvoke("check", config={"callbacks": [names]})

    assert "azure_attempt" in names.chains
    assert names.models == [DEPLOYMENT]


def test_run_metadata_and_attempt_count_follow_the_first_configured_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _azure_settings(monkeypatch)
    settings.llm.openai_api_key = SecretStr("dummy")

    assert provider.model_metadata(settings)["provider"] == "azure"
    assert provider.model_metadata(settings)["model"] == DEPLOYMENT
    assert provider.fallback_count(settings) == 3


@pytest.mark.asyncio
async def test_health_accepts_azure_as_the_configured_chat_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    async def embedding_available(*, model: str | None = None) -> bool:
        return True

    async def langsmith_status() -> dict[str, bool]:
        return {"connected": False}

    class Store:
        count = 0

    monkeypatch.setattr(main.settings, "llm", _azure_settings(monkeypatch).llm)
    monkeypatch.setattr(main.settings.llm, "aws_region", None)
    monkeypatch.setattr(main, "check_ollama_health", embedding_available)
    monkeypatch.setattr(main, "check_langsmith_connection", langsmith_status)
    monkeypatch.setattr(main, "get_vector_store", lambda: Store())

    response = await main.health()

    assert response.status_code == 200
