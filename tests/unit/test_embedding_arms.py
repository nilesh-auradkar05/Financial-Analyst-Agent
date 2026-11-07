"""Trace: docs/test-plan.md §21, embedding provider switch and per-arm indexes (CM-5a; ADR-0011
amendment 2026-10-11).

The four embedding arms (Ollama Qwen3, Bedrock Cohere Embed v4, Bedrock Titan V2, Azure
`text-embedding-3-small`) are driven through `get_embeddings()` and the settings that select them.
The real provider clients run against one loopback fake that serves the three wire formats and
computes every vector from the request (tests/unit/fake_embeddings.py). Settings come from
environment variables only (tests/embedding_arms.py). No network beyond 127.0.0.1, dummy credentials.
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
    OLLAMA,
    SECRET_VALUES,
    Arm,
    embeddings_api,
    isolate_environment,
    use_embedding_settings,
)
from tests.rag.helpers import SizedFakeEmbedder, find_backend_objects
from tests.unit.fake_embeddings import (
    OLLAMA_PATH,
    FakeEmbeddingProviders,
    vector_for,
)

UNREACHABLE_OPENSEARCH = "http://127.0.0.1:1"
PROVIDER_MODULES = (
    "opensearchpy",
    "boto3",
    "botocore",
    "openai",
    "ollama",
    "httpx",
    "langchain_aws",
    "langchain_openai",
    "langchain_ollama",
)
DIMENSION_FIELD = {"bedrock": "dimensions", "azure": "dimensions"}
CREDENTIAL = {"ollama": "", "bedrock": AWS_ACCESS_KEY_ID, "azure": AZURE_API_KEY}


@pytest.fixture
def providers() -> Iterator[FakeEmbeddingProviders]:
    server = FakeEmbeddingProviders().start()
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
    return "output_dimension" if arm is BEDROCK_COHERE else DIMENSION_FIELD[arm.provider]


def test_default_is_qwen3_on_ollama_with_the_sec_filings_collection(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Default unchanged."""
    select(OLLAMA)

    assert embeddings_api.embedding_identifier() == "ollama:qwen3-embedding:4b:2560"
    vectors = embeddings_api.get_embeddings().embed_documents(["alpha"])
    (request, *_) = providers.requests_to("ollama")
    assert request.path == OLLAMA_PATH
    assert request.body["model"] == "qwen3-embedding:4b"
    assert len(vectors[0]) == 2560
    assert embeddings_api.embedding_index_name() == "sec_filings"
    store = OpenSearchVectorStore(url=UNREACHABLE_OPENSEARCH, embeddings=SizedFakeEmbedder(2560))
    assert store.index_name == "sec_filings"


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

    assert embeddings_api.embedding_identifier().endswith(f":{dimension}")
    embeddings = embeddings_api.get_embeddings()
    documents = embeddings.embed_documents(texts)
    query = embeddings.embed_query("a question about the filing")

    assert providers.requests
    for request in providers.requests:
        assert request.body[_dimension_field(arm)] == dimension
    assert [len(vector) for vector in documents] == [dimension] * len(texts)
    assert len(query) == dimension
    input_type = "search_document" if arm is BEDROCK_COHERE else ""
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


def test_query_and_document_embeddings_of_the_same_text_use_different_input_types(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Query and document conventions."""
    select(BEDROCK_COHERE)
    assert embeddings_api.embedding_identifier() == BEDROCK_COHERE.identifier
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
    pytest.param(OLLAMA, "sec_filings", id="ollama-qwen3"),
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
    """test-plan §21: One index per arm (derived from the identifier; the default keeps sec_filings)."""
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
    select(OLLAMA)
    default_based = embeddings_api.embedding_index_name("other_base")

    assert len(set(names.values())) == len(names)
    assert names["cohere-512"] == "sec_filings__bedrock-cohere-embed-v4-0-512"
    assert based == "other_base__bedrock-cohere-embed-v4-0-1024"
    assert default_based == "other_base"


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
