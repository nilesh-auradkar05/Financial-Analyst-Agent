"""Trace: docs/test-plan.md §21, embedding provider switch and per-arm indexes (CM-5a; ADR-0011
amendment 2026-10-11).

The four embedding arms (Foundry Cohere Embed v5 fast, Bedrock Cohere Embed v4, Bedrock Titan V2,
Azure `text-embedding-3-small`) are driven through `get_embeddings()` and the settings that select
them. The real provider clients run against one loopback fake that serves the wire formats and
computes every vector from the request (tests/unit/fake_embeddings.py). Settings come from
environment variables only (tests/embedding_arms.py). No network beyond 127.0.0.1, dummy credentials.
CM-6e (2026-11-09): Ollama Qwen3 is gone; the default arm is Foundry Cohere Embed v5 fast, and every
arm, the default included, has an index named from its identifier. The Foundry arm paces its
requests, so every test here runs on the fake clock (tests/conftest.py `fake_clock`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.config import Settings
from tests.embedding_arms import (
    ALL_DUMMY_SECRETS,
    ARMS,
    AWS_ACCESS_KEY_ID,
    AZURE_API_KEY,
    AZURE_SMALL,
    BEARER_TOKEN,
    BEDROCK_COHERE,
    BEDROCK_TITAN,
    CLOUD_ARMS,
    FOUNDRY_COHERE,
    SECRET_VALUES,
    Arm,
    embeddings_api,
    isolate_environment,
    use_embedding_settings,
)
from tests.rag.helpers import SizedFakeEmbedder, find_backend_objects
from tests.unit.fake_embeddings import (
    FOUNDRY_PATH,
    FakeClock,
    FakeEmbeddingProviders,
    vector_for,
)

UNREACHABLE_OPENSEARCH = "http://127.0.0.1:1"
PROVIDER_MODULES = (
    "opensearchpy",
    "boto3",
    "botocore",
    "openai",
    "httpx",
    "langchain_aws",
    "langchain_openai",
)
DIMENSION_FIELD = {"bedrock": "dimensions", "azure": "dimensions"}
CREDENTIAL = {
    "foundry_cohere": AZURE_API_KEY,
    "bedrock": AWS_ACCESS_KEY_ID,
    "azure": AZURE_API_KEY,
}
COHERE_STYLE_ARMS = (FOUNDRY_COHERE, BEDROCK_COHERE)


@pytest.fixture(autouse=True)
def clock(fake_clock: FakeClock) -> FakeClock:
    return fake_clock


@pytest.fixture
def providers(clock: FakeClock) -> Iterator[FakeEmbeddingProviders]:
    server = FakeEmbeddingProviders(clock).start()
    yield server
    server.stop()


Select = Callable[..., None]


@pytest.fixture
def select(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, providers: FakeEmbeddingProviders
) -> Select:
    """Select an arm by environment alone, with its client routed to the loopback fake."""

    def choose(arm: Arm, **overrides: str) -> None:
        use_embedding_settings(monkeypatch, tmp_path, {**arm.routed_to(providers.url), **overrides})

    return choose


def _dimension_field(arm: Arm) -> str:
    return "output_dimension" if arm in COHERE_STYLE_ARMS else DIMENSION_FIELD[arm.provider]


def test_default_is_cohere_embed_v5_fast_on_foundry_with_its_own_collection(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Default arm (no embedding setting at all selects it)."""
    select(FOUNDRY_COHERE)

    assert embeddings_api.embedding_identifier() == "foundry_cohere:cohere-embed-v5-fast:1024"
    vectors = embeddings_api.get_embeddings().embed_documents(["alpha"])
    (request, *_) = providers.requests_to("foundry_cohere")
    assert request.path == FOUNDRY_PATH
    assert request.body["model"] == "cohere-embed-v5-fast"
    assert len(vectors[0]) == 1024
    expected = "sec_filings__foundry-cohere-cohere-embed-v5-fast-1024"
    assert embeddings_api.embedding_index_name() == expected
    store = OpenSearchVectorStore(url=UNREACHABLE_OPENSEARCH, embeddings=SizedFakeEmbedder(1024))
    assert store.index_name == expected
    assert store.index_name != "sec_filings"


def test_default_settings_name_the_foundry_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """test-plan §21: Default arm (the provider default; deployment and dimensions are checked
    through the identifier in the test above)."""
    isolate_environment(monkeypatch, tmp_path)

    loaded = Settings()

    assert loaded.embedding.provider == "foundry_cohere"


def test_foundry_arm_loads_without_any_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """test-plan §21: Unknown or incomplete configuration (the foundry_cohere arm loads without
    credentials; the missing setting is reported on first use)."""
    isolate_environment(monkeypatch, tmp_path, {"EMBEDDING_PROVIDER": "foundry_cohere"})

    loaded = Settings()

    assert loaded.embedding.provider == "foundry_cohere"
    assert not loaded.llm.azure_openai_endpoint
    assert not loaded.llm.azure_api_key


def test_the_settings_object_has_no_ollama_section() -> None:
    """test-plan §21 / §25: Ollama is removed; its settings class and section are gone."""
    import app.config

    assert not hasattr(app.config, "OllamaSettings")
    assert not hasattr(app.config.settings, "ollama")
    assert "ollama" not in Settings.model_fields


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
def test_each_arm_is_selected_by_settings_alone(
    select: Select, providers: FakeEmbeddingProviders, arm: Arm
) -> None:
    """test-plan §21: Provider selected by configuration."""
    select(arm)

    assert embeddings_api.embedding_identifier() == arm.identifier
    embeddings = embeddings_api.get_embeddings()
    documents = embeddings.embed_documents(["alpha filing", "beta filing"])
    query = embeddings.embed_query("gamma filing")
    assert {request.provider for request in providers.requests} == {arm.provider}
    assert {request.model for request in providers.requests} == {arm.model}
    assert {request.credential for request in providers.requests} == {CREDENTIAL[arm.provider]}
    assert [len(vector) for vector in documents] == [arm.dimension] * 2
    assert len(query) == arm.dimension
    # The helpers callers above the boundary use are unchanged and return plain floats.
    embedded = embeddings_api.embed_texts(["delta filing"])
    asked = embeddings_api.embed_query("epsilon filing")
    assert len(embedded[0]) == len(asked) == arm.dimension
    assert all(type(value) is float for value in [*embedded[0], *asked])


def _complete_env(arm: Arm) -> dict[str, str]:
    env = {**arm.env, **ALL_DUMMY_SECRETS, "AZURE_API_KEY": AZURE_API_KEY}
    env["AZURE_OPENAI_ENDPOINT"] = "https://embed-test.openai.azure.com"
    return env


REJECTED_SETTINGS = [
    pytest.param(
        BEDROCK_COHERE,
        {"EMBEDDING_PROVIDER": "openai_direct"},
        (),
        "EMBEDDING_PROVIDER",
        id="unknown-provider",
    ),
    pytest.param(
        BEDROCK_COHERE,
        {"EMBEDDING_PROVIDER": "ollama"},
        (),
        "EMBEDDING_PROVIDER",
        id="ollama-is-rejected",
    ),
    pytest.param(
        BEDROCK_COHERE, {}, ("EMBEDDING_MODEL",), "EMBEDDING_MODEL", id="bedrock-no-model"
    ),
    pytest.param(
        BEDROCK_COHERE,
        {},
        ("EMBEDDING_DIMENSIONS",),
        "EMBEDDING_DIMENSIONS",
        id="bedrock-no-dimensions",
    ),
    pytest.param(
        AZURE_SMALL, {}, ("EMBEDDING_MODEL",), "EMBEDDING_MODEL", id="azure-no-deployment"
    ),
    pytest.param(
        AZURE_SMALL, {}, ("EMBEDDING_DIMENSIONS",), "EMBEDDING_DIMENSIONS", id="azure-no-dimensions"
    ),
    pytest.param(
        AZURE_SMALL, {}, ("AZURE_OPENAI_ENDPOINT",), "AZURE_OPENAI_ENDPOINT", id="azure-no-endpoint"
    ),
    pytest.param(AZURE_SMALL, {}, ("AZURE_API_KEY",), "AZURE_API_KEY", id="azure-no-key"),
    pytest.param(
        BEDROCK_TITAN,
        {"EMBEDDING_DIMENSIONS": "0"},
        (),
        "EMBEDDING_DIMENSIONS",
        id="zero-dimensions",
    ),
    pytest.param(
        BEDROCK_TITAN,
        {"EMBEDDING_DIMENSIONS": "-1024"},
        (),
        "EMBEDDING_DIMENSIONS",
        id="negative-dimensions",
    ),
    pytest.param(
        BEDROCK_TITAN,
        {"EMBEDDING_DIMENSIONS": "wide"},
        (),
        "EMBEDDING_DIMENSIONS",
        id="non-numeric-dimensions",
    ),
]


@pytest.mark.parametrize(("arm", "changes", "removed", "named"), REJECTED_SETTINGS)
def test_unknown_or_incomplete_configuration_is_rejected_when_settings_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm: Arm,
    changes: dict[str, str],
    removed: tuple[str, ...],
    named: str,
) -> None:
    """test-plan §21: Unknown or incomplete configuration."""
    env = {
        name: value
        for name, value in {**_complete_env(arm), **changes}.items()
        if name not in removed
    }
    isolate_environment(monkeypatch, tmp_path, env)

    with pytest.raises(ValidationError) as rejected:
        Settings()

    message = str(rejected.value)
    assert named in message
    for secret in SECRET_VALUES:
        assert secret not in message


@pytest.mark.parametrize("dimension", [1024, 256])
@pytest.mark.parametrize("arm", CLOUD_ARMS, ids=lambda arm: arm.name)
def test_cloud_arm_asks_its_model_for_the_configured_dimension(
    select: Select, providers: FakeEmbeddingProviders, arm: Arm, dimension: int
) -> None:
    """test-plan §21: Requested dimension honoured."""
    select(arm, EMBEDDING_DIMENSIONS=str(dimension))
    texts = ["first filing text", "second filing text", "third filing text"]

    assert embeddings_api.embedding_identifier() == f"{arm.identifier.rpartition(':')[0]}:{dimension}"
    embeddings = embeddings_api.get_embeddings()
    documents = embeddings.embed_documents(texts)
    query = embeddings.embed_query("a question about the filing")

    assert providers.requests
    for request in providers.requests:
        assert request.body[_dimension_field(arm)] == dimension
    assert [len(vector) for vector in documents] == [dimension] * len(texts)
    assert len(query) == dimension
    input_type = "search_document" if arm in COHERE_STYLE_ARMS else ""
    for vector, text in zip(documents, texts):
        assert vector == pytest.approx(vector_for(arm.model, text, dimension, input_type), abs=1e-6)


@pytest.mark.parametrize("source", ["dotenv-file", "legacy-env-name"])
@pytest.mark.parametrize("arm", [BEDROCK_COHERE, BEDROCK_TITAN], ids=lambda arm: arm.name)
def test_bedrock_arm_authenticates_with_the_bearer_token_that_settings_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    providers: FakeEmbeddingProviders,
    arm: Arm,
    source: str,
) -> None:
    """test-plan §21: Provider selected by configuration (the bearer-token credential the rest of
    the app reads from settings, whether it sits in a dotenv file or under the legacy name)."""
    env = {
        name: value
        for name, value in arm.routed_to(providers.url).items()
        if name not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    }
    if source == "dotenv-file":
        (tmp_path / ".env").write_text(f"AWS_BEARER_TOKEN_BEDROCK={BEARER_TOKEN}\n")
    else:
        env["AWS_BEARER_TOKEN"] = BEARER_TOKEN
    use_embedding_settings(monkeypatch, tmp_path, env)
    # Only settings know the token: boto3 would see it only under its canonical environment name.
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ
    assert embeddings_api.embedding_identifier() == arm.identifier

    vectors = embeddings_api.get_embeddings().embed_documents(["alpha filing"])

    (request, *_) = providers.requests_to("bedrock")
    assert (request.scheme, request.credential) == ("bearer", BEARER_TOKEN)
    assert BEARER_TOKEN not in request.headers.values()
    assert [len(vector) for vector in vectors] == [arm.dimension]


@pytest.mark.parametrize("arm", COHERE_STYLE_ARMS, ids=lambda arm: arm.name)
def test_query_and_document_embeddings_of_the_same_text_use_different_input_types(
    select: Select, providers: FakeEmbeddingProviders, arm: Arm
) -> None:
    """test-plan §21: Query and document conventions."""
    select(arm)
    assert embeddings_api.embedding_identifier() == arm.identifier
    embeddings = embeddings_api.get_embeddings()
    text = "supply chain concentration risk"

    document = embeddings.embed_documents([text])[0]
    query = embeddings.embed_query(text)

    document_request, query_request = providers.requests
    assert document_request.body["texts"] == query_request.body["texts"] == [text]
    assert document_request.body["input_type"] == "search_document"
    assert query_request.body["input_type"] == "search_query"
    assert document != query


EXPECTED_INDEX_NAMES = [
    pytest.param(
        FOUNDRY_COHERE, "sec_filings__foundry-cohere-cohere-embed-v5-fast-1024", id="foundry-cohere"
    ),
    pytest.param(
        BEDROCK_COHERE, "sec_filings__bedrock-cohere-embed-v4-0-1024", id="bedrock-cohere"
    ),
    pytest.param(
        BEDROCK_TITAN, "sec_filings__bedrock-amazon-titan-embed-text-v2-0-1024", id="bedrock-titan"
    ),
    pytest.param(AZURE_SMALL, "sec_filings__azure-openai-embed-3-small-1024", id="azure-small"),
]


@pytest.mark.parametrize(("arm", "expected"), EXPECTED_INDEX_NAMES)
def test_each_arm_names_its_own_stable_collection(select: Select, arm: Arm, expected: str) -> None:
    """test-plan §21: One index per arm (derived from the identifier for every arm, the default included)."""
    select(arm)
    first = embeddings_api.embedding_index_name()
    select(arm)
    second = embeddings_api.embedding_index_name()

    store = OpenSearchVectorStore(
        url=UNREACHABLE_OPENSEARCH, embeddings=SizedFakeEmbedder(arm.dimension)
    )

    assert first == second == expected
    assert store.index_name == expected


def test_arms_never_share_a_collection_and_a_base_name_is_kept(select: Select) -> None:
    """test-plan §21: One index per arm (distinct per arm and per dimension; base name respected)."""
    names: dict[str, str] = {}
    for arm in ARMS:
        select(arm)
        names[arm.name] = embeddings_api.embedding_index_name()
    select(BEDROCK_COHERE, EMBEDDING_DIMENSIONS="512")
    names["cohere-512"] = embeddings_api.embedding_index_name()
    select(BEDROCK_COHERE)
    based = embeddings_api.embedding_index_name("other_base")
    select(FOUNDRY_COHERE)
    default_based = embeddings_api.embedding_index_name("other_base")

    assert len(set(names.values())) == len(names)
    assert names["cohere-512"] == "sec_filings__bedrock-cohere-embed-v4-0-512"
    assert based == "other_base__bedrock-cohere-embed-v4-0-1024"
    assert default_based == "other_base__foundry-cohere-cohere-embed-v5-fast-1024"


def test_an_explicit_index_name_still_wins_verbatim(
    select: Select, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test-plan §21: One index per arm (explicit names are not rewritten)."""
    select(BEDROCK_COHERE)
    embeddings = SizedFakeEmbedder(1024)

    explicit = OpenSearchVectorStore(
        index_name="Explicit_Index", url=UNREACHABLE_OPENSEARCH, embeddings=embeddings
    )
    monkeypatch.setenv("OPENSEARCH_INDEX_NAME", "From_Env_Index")
    from_env = OpenSearchVectorStore(url=UNREACHABLE_OPENSEARCH, embeddings=embeddings)

    assert explicit.index_name == "Explicit_Index"
    assert from_env.index_name == "From_Env_Index"


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
def test_no_provider_client_is_reachable_from_outside_the_retrieval_boundary(
    select: Select, arm: Arm
) -> None:
    """test-plan §21: Provider client stays inside the boundary (runtime half; the import half
    is tests/unit/test_layering.py)."""
    select(arm)
    assert embeddings_api.embedding_identifier() == arm.identifier

    store = OpenSearchVectorStore(url=UNREACHABLE_OPENSEARCH)
    surface: dict[str, Any] = {
        "store": store,
        "embed_texts": embeddings_api.embed_texts(["alpha"]),
        "embed_query": embeddings_api.embed_query("alpha"),
    }

    leaks = [
        leak
        for name, value in surface.items()
        for leak in find_backend_objects(value, name, modules=PROVIDER_MODULES)
    ]
    assert leaks == []
    assert not hasattr(store, "client")
