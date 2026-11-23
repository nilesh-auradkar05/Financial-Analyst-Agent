"""Trace: docs/test-plan.md §25, "Rerank pacing (S5.3)".

Within one process, consecutive rerank calls start at least 60/18 s apart and the estimated tokens
(ceil(chars / 4) of the query plus the documents) of the calls started in any 60 s window total at
most 18,000, under the deployment's 20 requests and 20,000 tokens per minute. An idle process is
not delayed, and rerank and embedding calls are paced in separate windows.

Time is the fake clock (tests/conftest.py `fake_clock`): sleeping is free and advances
`time.monotonic`. Each request's start is the fake clock at the moment it reaches the loopback
rerank endpoint (tests/unit/fake_foundry.py); the tolerance covers loopback jitter. No network
beyond 127.0.0.1, dummy credentials.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from app.components.retrieval.reranked_hybrid_retrieve import FoundryCohereReranker
from app.config import settings
from tests.embedding_arms import embeddings_api
from tests.rerank_api import rerank_api
from tests.unit.fake_embeddings import FakeClock
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, FakeFoundry, ReceivedRequest
from tests.unit.fake_retrieval_store import FakeRetrievalStore, make_point

INTERVAL = 60 / 18  # seconds between request starts: 18 per minute, under the 20 RPM ceiling
TOLERANCE = 0.25
TOKENS_PER_MINUTE = 18_000
QUERY = "supply chain supplier risks"


@pytest.fixture(autouse=True)
def clock(fake_clock: FakeClock) -> FakeClock:
    return fake_clock


def _tokens(text: str) -> int:
    """The documented estimate: ceil(characters / 4)."""
    return -(-len(text) // 4)


def _request_tokens(request: ReceivedRequest) -> int:
    return _tokens(request.body["query"]) + sum(_tokens(text) for text in request.body["documents"])


def _documents(count: int, characters: int) -> list[str]:
    """Distinct passages of exactly `characters` characters."""
    return [f"{n:05d} supply ".ljust(characters, "x") for n in range(count)]


def _gaps(requests: list[ReceivedRequest]) -> list[float]:
    return [later.arrived_at - earlier.arrived_at for earlier, later in zip(requests, requests[1:])]


def _search_once(documents: list[str]) -> None:
    """One `search_reranked` call whose candidates are exactly `documents`, on the default reranker."""
    store = FakeRetrievalStore(
        [make_point(f"chunk-{n:03d}", text, "risk_factors") for n, text in enumerate(documents)]
    )
    rerank_api.search_reranked(
        store, query=QUERY, ticker="AAPL", filing_type="10-K", top_k=3, candidates="dense"
    )


@pytest.mark.parametrize("fresh_instance", [False, True], ids=["one-reranker", "reranker-per-call"])
def test_consecutive_calls_start_at_least_one_interval_apart(
    foundry_rerank: FakeFoundry, fresh_instance: bool
) -> None:
    """test-plan §25: Rerank pacing (consecutive calls start >= 60/18 s apart, process-wide)."""
    shared = FoundryCohereReranker()
    documents = _documents(4, 100)

    for _ in range(6):
        (shared if not fresh_instance else FoundryCohereReranker()).score(QUERY, documents)

    requests = foundry_rerank.requests
    assert len(requests) == 6
    for gap in _gaps(requests):
        assert gap >= INTERVAL - TOLERANCE


def test_searches_through_the_default_reranker_are_paced_too(foundry_rerank: FakeFoundry) -> None:
    """test-plan §25: Rerank pacing (the benchmark runner builds a reranker per case; spacing holds
    across `search_reranked` calls)."""
    documents = _documents(6, 200)

    for _ in range(5):
        _search_once(documents)

    requests = foundry_rerank.requests
    assert len(requests) == 5
    for gap in _gaps(requests):
        assert gap >= INTERVAL - TOLERANCE


def test_a_call_on_an_idle_process_is_not_delayed(
    foundry_rerank: FakeFoundry, clock: FakeClock
) -> None:
    """test-plan §25: Rerank pacing (an idle process is not delayed)."""
    FoundryCohereReranker().score(QUERY, _documents(4, 100))

    assert clock.slept == 0


def test_a_call_after_a_busy_minute_is_not_delayed_once_the_window_has_passed(
    foundry_rerank: FakeFoundry, clock: FakeClock
) -> None:
    """test-plan §25: Rerank pacing (an idle process, after calls that filled the window)."""
    heavy = _documents(50, 400)  # 5,000 estimated tokens per call
    for _ in range(3):
        FoundryCohereReranker().score(QUERY, heavy)
    clock.advance(61)
    slept_before = clock.slept

    FoundryCohereReranker().score(QUERY, heavy)

    assert clock.slept == slept_before


def test_estimated_tokens_in_any_60_second_window_stay_within_18000(
    foundry_rerank: FakeFoundry, clock: FakeClock
) -> None:
    """test-plan §25: Rerank pacing (token budget over any 60 s window)."""
    heavy = _documents(50, 400)  # 5,000 estimated tokens per call, three per window

    for _ in range(8):
        FoundryCohereReranker().score(QUERY, heavy)

    requests = foundry_rerank.requests
    assert len(requests) == 8
    assert sum(_request_tokens(request) for request in requests) > 2 * TOKENS_PER_MINUTE
    assert requests[-1].arrived_at - requests[0].arrived_at >= 60 - TOLERANCE
    window = 60 - TOLERANCE  # arrival times carry loopback jitter
    for first in requests:
        started_with = sum(
            _request_tokens(other)
            for other in requests
            if first.arrived_at <= other.arrived_at < first.arrived_at + window
        )
        assert started_with <= TOKENS_PER_MINUTE, first.arrived_at


@pytest.fixture
def rerank_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeFoundry]:
    """A rerank server this test points the settings at on demand (the embedding fixture owns the
    endpoint setting otherwise)."""
    server = FakeFoundry().start()
    monkeypatch.setattr(settings.llm, "azure_foundry_rerank_model", RERANK_DEPLOYMENT)
    yield server
    server.stop()


def test_embedding_calls_do_not_hold_back_a_rerank_call(
    monkeypatch: pytest.MonkeyPatch,
    foundry_embed,
    rerank_endpoint: FakeFoundry,
    clock: FakeClock,
) -> None:
    """test-plan §25: Rerank pacing (rerank and embedding calls are paced in separate windows):
    after a filing's worth of embedding requests, a rerank call starts at once."""
    embeddings_api.get_embeddings().embed_documents(_documents(40, 1000))
    monkeypatch.setattr(settings.llm, "azure_openai_endpoint", rerank_endpoint.url)
    monkeypatch.setattr(settings.llm, "azure_api_key", SecretStr("test-key"))
    slept_before = clock.slept

    FoundryCohereReranker().score(QUERY, _documents(50, 400))

    assert len(rerank_endpoint.requests) == 1
    assert clock.slept == slept_before


def test_rerank_calls_do_not_hold_back_an_embedding_call(
    monkeypatch: pytest.MonkeyPatch,
    foundry_embed,
    rerank_endpoint: FakeFoundry,
    clock: FakeClock,
) -> None:
    """test-plan §25: Rerank pacing (separate windows): after rerank calls that filled their
    window, an embedding query starts at once."""
    embedding_endpoint = foundry_embed.url
    embedding_key = settings.llm.azure_api_key
    monkeypatch.setattr(settings.llm, "azure_openai_endpoint", rerank_endpoint.url)
    monkeypatch.setattr(settings.llm, "azure_api_key", SecretStr("test-key"))
    for _ in range(3):
        FoundryCohereReranker().score(QUERY, _documents(50, 400))
    monkeypatch.setattr(settings.llm, "azure_openai_endpoint", embedding_endpoint)
    monkeypatch.setattr(settings.llm, "azure_api_key", embedding_key)
    slept_before = clock.slept

    vector = embeddings_api.embed_query("what are the main supply chain risks")

    assert len(vector) > 0
    assert clock.slept == slept_before
