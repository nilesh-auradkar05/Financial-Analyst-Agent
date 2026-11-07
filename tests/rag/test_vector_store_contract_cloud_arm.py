"""The RetrievalStore contract suite (test-plan §4, §5) on a cloud-arm index (test-plan §21; CM-5a,
layer C).

The suite is reused, not copied: every test of `test_vector_store_contract` runs here with the
settings of the Bedrock Cohere Embed v4 arm (1024-d) and the first store opened on the collection
name that arm derives from its embedding identifier. That collection is deleted before and after
each test, so runs never see each other's data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from tests.embedding_arms import BEDROCK_COHERE, embeddings_api, use_embedding_settings
from tests.rag.helpers import FakeEmbedder, SizedFakeEmbedder, raw_request
from tests.rag.test_vector_store_contract import *  # noqa: F403 - re-run the whole suite here

NO_ENDPOINT = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def settings_name_the_cloud_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_embedding_settings(monkeypatch, tmp_path, BEDROCK_COHERE.routed_to(NO_ENDPOINT))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return SizedFakeEmbedder(BEDROCK_COHERE.dimension)


@pytest.fixture
def make_store(
    opensearch_url: str,
    embedder: FakeEmbedder,
    new_index: Callable[[], str],
) -> Iterator[Callable[..., OpenSearchVectorStore]]:
    """Stores on the test container: the first one opened gets the arm's derived collection name."""
    arm_index = embeddings_api.embedding_index_name()
    assert arm_index == BEDROCK_COHERE.index_name
    raw_request(opensearch_url, "DELETE", f"/{arm_index}")
    opened: list[str] = []

    def build(index: str | None = None) -> OpenSearchVectorStore:
        name = index or (arm_index if not opened else new_index())
        opened.append(name)
        return OpenSearchVectorStore(
            index_name=name, url=opensearch_url, timeout_s=10, embeddings=embedder
        )

    yield build

    raw_request(opensearch_url, "DELETE", f"/{arm_index}")
