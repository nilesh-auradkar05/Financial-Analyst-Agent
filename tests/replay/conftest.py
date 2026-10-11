"""Fixtures for the zero-network replay suite (docs/test-plan.md section 13).

Scope: this directory only. Every fixture that changes global behaviour (socket guard,
tracing switch) is autouse here and nowhere else.
"""

from __future__ import annotations

import itertools
import os
import socket
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from tests.replay.support import (
    BuiltRelease,
    FakeChat,
    LiveToolUsedError,
    NetworkAccessError,
    build_release,
)

TRACING_ENV = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)
TRACING_KEYS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")


@dataclass
class NetworkGuard:
    """Handle on the active socket guard.

    `attempts` records every blocked connect. A production `except Exception` could swallow
    NetworkAccessError, so the fixture also fails the test at teardown if any attempt was
    recorded and not explicitly cleared by the guard's own self-check.
    `real_connect` is the unpatched connect; only the self-check uses it.
    """

    attempts: list[str]
    real_connect: Callable[..., Any]


@pytest.fixture(autouse=True)
def network_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkGuard]:
    """Any socket connect / connect_ex / create_connection raises NetworkAccessError.

    AF_UNIX connects are left alone: they never leave the machine and are the family
    asyncio and anyio use for their internal wake-up channels. Everything else (IPv4,
    IPv6, loopback included) is refused.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    attempts: list[str] = []
    details: list[str] = []

    def refuse(address: object) -> NoReturn:
        attempts.append(repr(address))
        frames = itertools.islice(traceback.walk_stack(None), 1, 15)
        where = " <- ".join(
            f"{frame.f_code.co_filename.rsplit('/', 2)[-1]}:{lineno}:{frame.f_code.co_name}"
            for frame, lineno in ((f, f.f_lineno) for f, _ in frames)
        )
        details.append(
            f"thread={threading.current_thread().name} address={address!r} stack={where}"
        )
        raise NetworkAccessError(f"network access attempted in replay suite: {address!r}")

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
            return
        refuse(address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family == socket.AF_UNIX:
            return real_connect_ex(self, address)
        refuse(address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        refuse(address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)

    guard = NetworkGuard(attempts=attempts, real_connect=real_connect)
    yield guard
    assert guard.attempts == [], f"network connect attempted during test: {details}"


def clear_langsmith_env_cache() -> None:
    """Drop langsmith's lru_cache on env lookups (its overloaded typing hides cache_clear)."""
    from langsmith import utils as langsmith_utils

    getattr(langsmith_utils.get_env_var, "cache_clear")()


@pytest.fixture(autouse=True)
def tracing_off() -> Iterator[None]:
    """Turn LangSmith/LangChain tracing off so background trace uploads never reach the guard.

    The env vars are set and restored by hand because langsmith caches env lookups in
    `get_env_var`; the cache is cleared on both sides so neither this scope nor later
    tests see a stale value. `tracing_context(enabled=False)` is the authoritative switch
    (langsmith checks it before the environment).
    """
    from langsmith import tracing_context

    from app.config import settings

    saved = {key: os.environ.get(key) for key in (*TRACING_ENV, *TRACING_KEYS)}
    saved_flag = settings.langsmith.tracing_v2
    saved_key = settings.langsmith.api_key
    for key in TRACING_ENV:
        os.environ[key] = "false"
    for key in TRACING_KEYS:
        os.environ.pop(key, None)
    settings.langsmith.tracing_v2 = False
    settings.langsmith.api_key = None
    clear_langsmith_env_cache()
    try:
        with tracing_context(enabled=False):
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        settings.langsmith.tracing_v2 = saved_flag
        settings.langsmith.api_key = saved_key
        clear_langsmith_env_cache()


@pytest.fixture
def release(tmp_path) -> BuiltRelease:
    """Default registered evidence_snapshot release (two tickers, all four source types)."""
    return build_release(tmp_path / "default")


@pytest.fixture
def make_release(tmp_path) -> Callable[..., BuiltRelease]:
    """Build further releases, each in its own snapshot root and registry."""
    counter = iter(range(1, 100))

    def factory(**kwargs: Any) -> BuiltRelease:
        return build_release(tmp_path / f"release-{next(counter)}", **kwargs)

    return factory


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> FakeChat:
    """Replace the memo model with a faithful in-process fake."""
    from app.agents import graph

    model = FakeChat()
    monkeypatch.setattr(graph, "get_llm", lambda _settings: model)
    return model


@pytest.fixture
def no_live_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every live evidence entry point raises if reached.

    Covers news search, stock fetch, filing retrieval / vector store, ingestion, the
    sentiment model and the embedding model, both where `app.agents.graph` imported them
    and at their home modules.
    """

    def forbidden(name: str) -> Callable[..., Any]:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise LiveToolUsedError(f"replay reached a live evidence path: {name}")

        return refuse

    from app.agents import graph
    from app.components.retrieval import embeddings, vector_store
    from app.services import sentiment
    from app.services.tools import stock_data_tool, web_search_tool

    targets: list[tuple[Any, str]] = [
        (graph, "search_company_news"),
        (graph, "get_stock_data"),
        (graph, "get_vector_store"),
        (graph, "ingest_10k_for_ticker"),
        (graph, "analyze_sentiment_batch"),
        (web_search_tool, "search_company_news"),
        (stock_data_tool, "get_stock_data"),
        (vector_store, "get_vector_store"),
        (sentiment, "analyze_sentiment_batch"),
        (embeddings, "get_embeddings"),
        (embeddings, "embed_texts"),
        (embeddings, "embed_query"),
        (embeddings, "embed_texts_async"),
        (embeddings, "embed_query_async"),
        (embeddings.EmbeddingsClient, "embed_single"),
        (embeddings.EmbeddingsClient, "embed_texts"),
        (embeddings.EmbeddingsClient, "embed_query"),
    ]
    for owner, attribute in targets:
        monkeypatch.setattr(
            owner, attribute, forbidden(f"{getattr(owner, '__name__', owner)}.{attribute}")
        )


# One filing chunk's time fields: (metadata["fetched_at"], filing_date); None omits a field.
ChunkTimes = tuple[str | None, str | None]
DEFAULT_FILING_TIMES: tuple[ChunkTimes, ...] = ((None, "2026-01-01"), (None, "2026-01-01"))


@pytest.fixture
def live_world(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> Callable[..., None]:
    """Install faithful in-process fakes for every live evidence entry point.

    The fakes compute their output from their inputs (ticker, query, count) and run
    through the real nodes and the real compiled graph. The returned installer takes:

    - `price_offset`: changes the live evidence deterministically;
    - `news_count`: articles the news source returns (0 = it returns nothing);
    - `stock_available`: False makes the market-data source fail;
    - `filing_times`: one `(fetched_at, filing_date)` pair per chunk the store returns for a
      query (chunk n of every result gets pair n); an empty sequence = nothing indexed.
    """
    from app.agents import graph

    def install(
        *,
        price_offset: float = 0.0,
        news_count: int = 3,
        stock_available: bool = True,
        filing_times: Sequence[ChunkTimes] = DEFAULT_FILING_TIMES,
    ) -> None:
        async def news(query: str, max_results: int = 10) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    title=f"{query} #{n}",
                    url=f"https://example.test/live/{n}",
                    source="Wire",
                    snippet=f"{query} item {n} reports revenue growth.",
                    published_date="2026-09-01",
                )
                for n in range(min(max_results, news_count))
            ]

        async def stock(ticker: str) -> SimpleNamespace:
            if not stock_available:
                raise RuntimeError(f"no market data for {ticker}")
            return SimpleNamespace(
                ticker=ticker,
                company_name=f"{ticker} Corporation",
                current_price=float(sum(map(ord, ticker))) + price_offset,
                price_change_percent=1.0,
                market_cap=1_000_000,
                pe_ratio=10.0,
                fifty_two_week_high=400.0,
                fifty_two_week_low=50.0,
                volume=100,
                dividend_yield=0.0,
                sector="Technology",
                industry="Software",
                recommendation="hold",
            )

        class Store:
            def search_by_ticker(
                self, query: str, ticker: str, n_results: int = 3, section: str | None = None
            ) -> SimpleNamespace:
                chunks = []
                for n, (fetched_at, filing_date) in enumerate(filing_times[:n_results]):
                    metadata = {"filing_type": "10-K", "company_name": f"{ticker} Inc."}
                    if fetched_at is not None:
                        metadata["fetched_at"] = fetched_at
                    chunks.append(
                        SimpleNamespace(
                            id=f"{ticker}:{query}:{n}",
                            text=f"{n} {query}: filing text for {ticker}.",
                            section="Business",
                            metadata=metadata,
                            filing_date=filing_date,
                            relevance_score=1 / (n + 1),
                        )
                    )
                return SimpleNamespace(chunks=chunks)

            def count_documents(self, filters: object = None) -> int:
                return len(filing_times)

        async def ingest_nothing(ticker: str) -> SimpleNamespace:
            return SimpleNamespace(success=False, error_code="filing_not_found")

        monkeypatch.setattr(graph, "ingest_10k_for_ticker", ingest_nothing)
        monkeypatch.setattr(graph, "search_company_news", news)
        monkeypatch.setattr(graph, "get_stock_data", stock)
        monkeypatch.setattr(graph, "get_vector_store", Store)
        monkeypatch.setattr(
            graph,
            "analyze_sentiment_batch",
            lambda texts: [
                SimpleNamespace(label="positive" if "growth" in text else "neutral")
                for text in texts
            ],
        )

    return install
