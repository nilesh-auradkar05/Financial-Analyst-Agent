from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.embedding_arms import use_embedding_settings
from tests.rag.helpers import FakeEmbedder, build_corpus, raw_request

DEFAULT_OPENSEARCH_TEST_URL = "http://127.0.0.1:19200"
START_COMMAND = (
    "docker compose -p financial-analyst-opensearch-test "
    "-f docker-compose.opensearch.yml up -d --wait"
)

StoreFactory = Callable[..., Any]


@pytest.fixture(scope="session")
def opensearch_url() -> str:
    """Local test container URL; an unreachable container fails the tests, never skips them."""
    url = os.environ.get("OPENSEARCH_TEST_URL", DEFAULT_OPENSEARCH_TEST_URL)
    try:
        with urllib.request.urlopen(url, timeout=3):
            pass
    except (urllib.error.URLError, OSError):
        pytest.fail(
            f"OpenSearch test container is not reachable at {url}; start it with: {START_COMMAND}",
            pytrace=False,
        )
    return url


@pytest.fixture(autouse=True)
def settings_describe_the_fake_embedder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Configuration names the 16-d fake (ollama:fake-hash-embed:16), so the embedding identifier
    and index dimension that settings dictate agree with the vectors the fake produces. Tests that
    exercise another arm call `use_embedding_settings` themselves."""
    use_embedding_settings(
        monkeypatch,
        tmp_path,
        {
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_MODEL": FakeEmbedder.model,
            "EMBEDDING_DIMENSIONS": str(FakeEmbedder.dimension),
        },
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def new_index(opensearch_url: str) -> Iterator[Callable[[], str]]:
    """Factory of unique index names, each deleted at teardown."""
    created: list[str] = []

    def create() -> str:
        name = f"cm2-test-{uuid.uuid4().hex}"
        created.append(name)
        return name

    yield create

    for name in created:
        raw_request(opensearch_url, "DELETE", f"/{name}")


@pytest.fixture
def index_name(new_index: Callable[[], str]) -> str:
    return new_index()


@pytest.fixture
def make_store(
    opensearch_url: str,
    embedder: FakeEmbedder,
    new_index: Callable[[], str],
) -> StoreFactory:
    """Build an ``OpenSearchVectorStore`` on the test container (a fresh index unless named)."""

    def build(index: str | None = None) -> Any:
        from app.components.retrieval.opensearch_store import OpenSearchVectorStore

        return OpenSearchVectorStore(
            index_name=index or new_index(),
            url=opensearch_url,
            timeout_s=10,
            embeddings=embedder,
        )

    return build


@pytest.fixture
def corpus() -> list:
    return build_corpus()
