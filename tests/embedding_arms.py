"""The four embedding arms of test-plan §21 and the settings isolation their tests share.

Settings reach the code under test only through environment variables, the way a deployment
supplies them. `use_embedding_settings` clears everything that could leak in from the developer's
shell or their `.env` file (the working directory moves to an empty folder, where no `.env` exists),
applies the arm's variables, then loads a fresh `Settings` into the singleton the retrieval boundary
reads. Every credential is a dummy string.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings, settings
from tests.unit.fake_embeddings import (
    AZURE_DEPLOYMENT,
    COHERE_MODEL,
    OLLAMA_MODEL,
    TITAN_MODEL,
)

# The retrieval module under test, resolved at run time: the embedding-switch functions it gains in
# CM-5a are looked up on the module, so a missing one fails the test that uses it, not the collection.
embeddings_api: Any = importlib.import_module("app.components.retrieval.embeddings")

AWS_ACCESS_KEY_ID = "dummy-access-key-id"
AWS_SECRET_ACCESS_KEY = "dummy-aws-secret-key-do-not-leak"
BEARER_TOKEN = "dummy-bearer-token-do-not-leak"
AZURE_API_KEY = "dummy-azure-api-key-do-not-leak"
SECRET_VALUES = (AWS_SECRET_ACCESS_KEY, BEARER_TOKEN, AZURE_API_KEY)

_CLEARED = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBED_BASE_URL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_API_KEY",
    "OPENSEARCH_INDEX_NAME",
    "CHROMA_COLLECTION_NAME",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEARER_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SESSION_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def isolate_environment(
    monkeypatch: pytest.MonkeyPatch, workdir: Path, env: dict[str, str] | None = None
) -> None:
    """Drop every variable the embedding settings or the AWS/OpenAI clients could read, then apply `env`."""
    monkeypatch.chdir(workdir)
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(workdir / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(workdir / "no-aws-credentials"))
    # A default Ollama may be listening on this host; no test may reach it by accident.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)


def use_embedding_settings(
    monkeypatch: pytest.MonkeyPatch, workdir: Path, env: dict[str, str] | None = None
) -> None:
    """Make the process-wide settings read exactly `env` (and no `.env` file, shell value or default profile)."""
    isolate_environment(monkeypatch, workdir, env)
    fresh = Settings()
    # `embedding` exists once the provider switch lands; until then only llm/ollama are swapped.
    for name in ("embedding", "llm", "ollama"):
        value = getattr(fresh, name, None)
        if value is not None:
            monkeypatch.setattr(settings, name, value)


@dataclass(frozen=True)
class Arm:
    """One embedding arm: how it is configured, how its provider names it on the wire, what it is called."""

    name: str
    provider: str  # which route of FakeEmbeddingProviders serves it
    model: str  # the model or deployment name the provider receives
    dimension: int
    identifier: str
    index_name: str
    env: dict[str, str] = field(default_factory=dict)

    def routed_to(self, url: str) -> dict[str, str]:
        """The arm's variables plus the ones that point its client at a loopback fake at `url`."""
        if self.provider == "ollama":
            return {**self.env, "OLLAMA_EMBED_BASE_URL": url, "OLLAMA_BASE_URL": url}
        if self.provider == "bedrock":
            return {
                **self.env,
                "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": url,
                "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
                "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            }
        return {**self.env, "AZURE_OPENAI_ENDPOINT": url, "AZURE_API_KEY": AZURE_API_KEY}

    def with_dimension(self, dimension: int) -> dict[str, str]:
        return {**self.env, "EMBEDDING_DIMENSIONS": str(dimension)}


OLLAMA = Arm(
    name="ollama-qwen3",
    provider="ollama",
    model=OLLAMA_MODEL,
    dimension=2560,
    identifier="ollama:qwen3-embedding:4b:2560",
    index_name="sec_filings",
)
BEDROCK_COHERE = Arm(
    name="bedrock-cohere-v4",
    provider="bedrock",
    model=COHERE_MODEL,
    dimension=1024,
    identifier="bedrock:cohere.embed-v4:0:1024",
    index_name="sec_filings__bedrock-cohere-embed-v4-0-1024",
    env={
        "EMBEDDING_PROVIDER": "bedrock",
        "EMBEDDING_MODEL": COHERE_MODEL,
        "EMBEDDING_DIMENSIONS": "1024",
    },
)
BEDROCK_TITAN = Arm(
    name="bedrock-titan-v2",
    provider="bedrock",
    model=TITAN_MODEL,
    dimension=1024,
    identifier="bedrock:amazon.titan-embed-text-v2:0:1024",
    index_name="sec_filings__bedrock-amazon-titan-embed-text-v2-0-1024",
    env={
        "EMBEDDING_PROVIDER": "bedrock",
        "EMBEDDING_MODEL": TITAN_MODEL,
        "EMBEDDING_DIMENSIONS": "1024",
    },
)
AZURE_SMALL = Arm(
    name="azure-text-embedding-3-small",
    provider="azure",
    model=AZURE_DEPLOYMENT,
    dimension=1024,
    identifier="azure_openai:embed-3-small:1024",
    index_name="sec_filings__azure-openai-embed-3-small-1024",
    env={
        "EMBEDDING_PROVIDER": "azure_openai",
        "EMBEDDING_MODEL": AZURE_DEPLOYMENT,
        "EMBEDDING_DIMENSIONS": "1024",
    },
)
ARMS = (OLLAMA, BEDROCK_COHERE, BEDROCK_TITAN, AZURE_SMALL)
CLOUD_ARMS = (BEDROCK_COHERE, BEDROCK_TITAN, AZURE_SMALL)

# Present in the environment of every incomplete-configuration case, so a secret echoed into an
# error message would be caught.
ALL_DUMMY_SECRETS = {
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_BEARER_TOKEN_BEDROCK": BEARER_TOKEN,
}
