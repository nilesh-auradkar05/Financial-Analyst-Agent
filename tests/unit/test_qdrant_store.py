from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from app.components.retrieval.vector_store import IndexDocument, SearchFilters


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


@dataclass
class FakeCountResponse:
    count: int


@dataclass
class FakeScoredPoint:
    id: str
    payload: dict[str, Any]
    score: float


@dataclass
class FakeQueryResponse:
    points: list[FakeScoredPoint]


class FakeQdrantClient:
    def __init__(self, *args: Any, **kwargs: Any):
        self.collection_created = False
        self.index_fields: list[str] = []
        self.points: list[Any] = []
        self.last_query_filter: Any = None
        self.last_count_filter: Any = None
        self.last_deleted_ids: list[Any] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_created

    def create_collection(self, **kwargs: Any) -> None:
        self.collection_created = True
        self.create_collection_kwargs = kwargs

    def create_payload_index(self, **kwargs: Any) -> None:
        self.index_fields.append(kwargs["field_name"])

    def upsert(self, **kwargs: Any) -> None:
        self.points.extend(kwargs["points"])

    def query_points(self, **kwargs: Any) -> FakeQueryResponse:
        self.last_query_filter = kwargs.get("query_filter")
        scored = [
            FakeScoredPoint(
                id=str(point.id),
                payload=point.payload,
                score=0.91,
            )
            for point in self.points
        ]
        return FakeQueryResponse(points=scored[: kwargs["limit"]])

    def count(self, **kwargs: Any) -> FakeCountResponse:
        self.last_count_filter = kwargs.get("count_filter")
        return FakeCountResponse(count=len(self.points))

    def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
        return ([], None)

    def delete(self, **kwargs: Any) -> None:
        self.last_deleted_ids = list(kwargs["points_selector"].points)

    def get_collection(self, **kwargs: Any) -> object:
        return object()


@pytest.fixture
def qdrant_store(monkeypatch: pytest.MonkeyPatch):
    import app.components.retrieval.qdrant_store as module

    client = FakeQdrantClient()
    monkeypatch.setattr(module, "QdrantClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(module, "get_embeddings", lambda: FakeEmbeddings())
    store = module.QdrantVectorStore(
        collection_name="test_sec_filings",
        url="http://qdrant-test:6333",
    )
    return store, client


def test_qdrant_add_documents_upserts_payload_with_original_retrieval_id(qdrant_store):
    store, client = qdrant_store

    written = store.add_documents(
        [
            IndexDocument(
                id="AAPL_10-K_2025-10-31_business_000",
                text="Apple business overview",
                metadata={
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "section_key": "business",
                    "filing_date": "2025-10-31",
                },
            )
        ]
    )

    assert written == 1
    assert client.collection_created is True
    assert set(client.index_fields) >= {"ticker", "filing_type", "section_key", "filing_date"}
    assert client.points[0].payload["_retrieval_id"] == "AAPL_10-K_2025-10-31_business_000"
    assert client.points[0].payload["text"] == "Apple business overview"
    assert client.points[0].payload["ticker"] == "AAPL"


def test_qdrant_search_returns_original_chunk_id_and_applies_filter(qdrant_store):
    store, client = qdrant_store
    store.add_documents(
        [
            IndexDocument(
                id="AAPL_10-K_2025-10-31_risk_000",
                text="Apple supply chain risk",
                metadata={"ticker": "AAPL", "section_key": "risk_factors"},
            )
        ]
    )

    result = store.search(
        "supply chain risk",
        filters=SearchFilters(ticker="aapl", section_key="risk_factors"),
        n_results=3,
    )

    assert result.has_results is True
    assert result.chunks[0].id == "AAPL_10-K_2025-10-31_risk_000"
    assert result.chunks[0].text == "Apple supply chain risk"
    assert result.chunks[0].metadata["ticker"] == "AAPL"
    assert client.last_query_filter is not None


def test_qdrant_count_uses_payload_filter(qdrant_store):
    store, client = qdrant_store
    store.add_documents(
        [
            IndexDocument(
                id="MSFT_10-K_2025-06-30_business_000",
                text="Microsoft cloud business",
                metadata={"ticker": "MSFT", "section_key": "business"},
            )
        ]
    )

    assert store.count_documents(SearchFilters(ticker="MSFT")) == 1
    assert client.last_count_filter is not None


def test_qdrant_point_id_mapping_is_deterministic_uuid_without_project_namespace_constant():
    import app.components.retrieval.qdrant_store as module

    first = module._to_point_id("AAPL_10-K_2025-10-31_business_000")
    second = module._to_point_id("AAPL_10-K_2025-10-31_business_000")

    assert first == second
    assert str(uuid.UUID(first)) == first
    assert not hasattr(module, "QDRANT_ID_NAMESPACE")


def test_qdrant_search_returns_empty_result_when_collection_is_missing(qdrant_store):
    store, client = qdrant_store
    client.collection_created = False

    result = store.search("business overview", filters=SearchFilters(ticker="AAPL"))

    assert result.has_results is False
    assert result.total_results == 0


def test_qdrant_count_returns_zero_when_collection_is_missing(qdrant_store):
    store, client = qdrant_store
    client.collection_created = False

    assert store.count_documents(SearchFilters(ticker="AAPL")) == 0
