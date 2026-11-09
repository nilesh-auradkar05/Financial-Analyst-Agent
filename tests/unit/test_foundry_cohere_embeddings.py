"""Trace: docs/test-plan.md §21, Foundry Cohere request shape, malformed reply, pacing and throttle
(CM-6e), plus the removed Ollama-era embedding symbols.

The default embedding arm is driven through `get_embeddings()` / `embed_texts` / `embed_query`
against the loopback Foundry Cohere fake of tests/unit/fake_embeddings.py, which computes vectors
from each request, enforces the 96-text limit and can be scripted to throttle or misbehave. Time
is the fake clock (tests/conftest.py `fake_clock`): sleeping is free and advances `time.monotonic`.
Request spacing is read from the fake clock at the moment each request reaches the server; the
tolerance covers loopback jitter between the client sending and the server recording. No network
beyond 127.0.0.1, dummy credentials.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents import graph
from app.agents.state import create_initial_state
from tests.embedding_arms import (
    AZURE_API_KEY,
    FOUNDRY_COHERE,
    embeddings_api,
    use_embedding_settings,
)
from tests.unit.fake_embeddings import (
    FOUNDRY_MAX_TEXTS,
    FOUNDRY_MODEL,
    FOUNDRY_PATH,
    LEAKY_BODY_MARKER,
    FakeClock,
    FakeEmbeddingProviders,
    ReceivedEmbeddingRequest,
    ScriptedReply,
    vector_for,
)

INTERVAL = 60 / 18  # seconds between request starts: 18 per minute, under the 20 RPM ceiling
TOLERANCE = 0.25
NO_ENDPOINT = "http://127.0.0.1:1"
Select = Callable[..., None]


@pytest.fixture(autouse=True)
def clock(fake_clock: FakeClock) -> FakeClock:
    return fake_clock


@pytest.fixture
def providers(clock: FakeClock) -> Iterator[FakeEmbeddingProviders]:
    server = FakeEmbeddingProviders(clock).start()
    yield server
    server.stop()


@pytest.fixture
def select(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, providers: FakeEmbeddingProviders
) -> Select:
    """Select the default arm by environment alone, routed to the fake (or to `endpoint`)."""

    def choose(*, remove: tuple[str, ...] = (), **overrides: str) -> None:
        env = {**FOUNDRY_COHERE.routed_to(providers.url), **overrides}
        for name in remove:
            env.pop(name)
        use_embedding_settings(monkeypatch, tmp_path, env)
        # Precondition: the default arm is the one selected, so no test below can reach another provider.
        assert embeddings_api.embedding_identifier() == FOUNDRY_COHERE.identifier

    return choose


def _foundry(providers: FakeEmbeddingProviders) -> list[ReceivedEmbeddingRequest]:
    return providers.requests_to("foundry_cohere")


def _gaps(requests: list[ReceivedEmbeddingRequest]) -> list[float]:
    return [later.arrived_at - earlier.arrived_at for earlier, later in zip(requests, requests[1:])]


def _texts(count: int) -> list[str]:
    return [f"risk factor passage number {n}" for n in range(count)]


def _embed(operation: str) -> list[list[float]]:
    embeddings = embeddings_api.get_embeddings()
    if operation == "documents":
        documents: list[list[float]] = embeddings.embed_documents(["first passage", "second passage"])
        return documents
    return [embeddings.embed_query("a question about supply risk")]


def _assert_error_is_clean(error: BaseException) -> None:
    assert AZURE_API_KEY not in str(error)
    assert LEAKY_BODY_MARKER not in str(error)


OPERATIONS = pytest.mark.parametrize("operation", ["documents", "query"])


# ---- public surface ------------------------------------------------------------------------


def test_removed_ollama_era_symbols_are_gone_and_the_typed_error_exists() -> None:
    """test-plan §21 Default arm / §25 No local model packages: the Ollama-only embedding client
    and helpers are removed; the typed embedding error is a RuntimeError; the helpers callers use stay."""
    for name in (
        "DEFAULT_EMBEDDING_IDENTIFIER",
        "EmbeddingsClient",
        "EmbeddingResult",
        "embed_texts_async",
        "embed_query_async",
    ):
        assert not hasattr(embeddings_api, name), name
    for name in ("get_embeddings", "embed_texts", "embed_query", "EmbeddingDimensionMismatchError"):
        assert hasattr(embeddings_api, name), name
    assert issubclass(embeddings_api.EmbeddingUnavailableError, RuntimeError)


def test_the_default_arm_is_a_langchain_embeddings_object(select: Select) -> None:
    """test-plan §21 Provider selected by configuration: `get_embeddings()` returns the Foundry
    Cohere adapter, a langchain `Embeddings`."""
    from langchain_core.embeddings import Embeddings

    select()

    embeddings = embeddings_api.get_embeddings()

    assert isinstance(embeddings, Embeddings)
    assert isinstance(embeddings, embeddings_api.FoundryCohereEmbeddings)


# ---- request shape -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "batches"),
    [(1, [1]), (96, [96]), (97, [96, 1]), (200, [96, 96, 8])],
    ids=["one", "limit", "over-limit", "three-batches"],
)
def test_documents_go_out_in_batches_of_at_most_96_with_the_document_input_type(
    select: Select, providers: FakeEmbeddingProviders, count: int, batches: list[int]
) -> None:
    """test-plan §21: Foundry Cohere request shape."""
    select()
    texts = _texts(count)

    vectors = embeddings_api.get_embeddings().embed_documents(texts)

    requests = _foundry(providers)
    assert FOUNDRY_MAX_TEXTS == 96
    assert [len(request.body["texts"]) for request in requests] == batches
    assert [text for request in requests for text in request.body["texts"]] == texts
    for request in requests:
        assert request.path == FOUNDRY_PATH
        assert request.body["model"] == "cohere-embed-v5-fast"
        assert request.body["input_type"] == "search_document"
        assert request.body["embedding_types"] == ["float"]
        assert request.body["output_dimension"] == 1024
        assert (request.scheme, request.credential) == ("api-key", AZURE_API_KEY)
        assert AZURE_API_KEY not in request.headers.values()
    assert len(vectors) == count
    for vector, text in zip(vectors, texts):
        assert vector == pytest.approx(vector_for(FOUNDRY_MODEL, text, 1024, "search_document"))


def test_a_query_goes_out_alone_with_the_query_input_type(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere request shape (queries)."""
    select()

    vector = embeddings_api.get_embeddings().embed_query("how concentrated is the supply chain")

    (request,) = _foundry(providers)
    assert request.path == FOUNDRY_PATH
    assert request.body["texts"] == ["how concentrated is the supply chain"]
    assert request.body["input_type"] == "search_query"
    assert request.body["embedding_types"] == ["float"]
    assert request.body["output_dimension"] == 1024
    assert vector == pytest.approx(
        vector_for(FOUNDRY_MODEL, "how concentrated is the supply chain", 1024, "search_query")
    )


def test_the_boundary_helpers_use_the_same_adapter(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere request shape (embed_texts and embed_query are unchanged for callers)."""
    select()

    documents = embeddings_api.embed_texts(["alpha filing", "beta filing"])
    query = embeddings_api.embed_query("gamma filing")

    first, second = _foundry(providers)
    assert first.body["input_type"] == "search_document"
    assert second.body["input_type"] == "search_query"
    assert [len(vector) for vector in documents] == [1024, 1024]
    assert len(query) == 1024


# ---- malformed reply -----------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["short_count", "wrong_length"])
@pytest.mark.parametrize("operation", ["documents", "query"])
def test_a_reply_with_the_wrong_vector_count_or_length_is_refused(
    select: Select, providers: FakeEmbeddingProviders, operation: str, shape: str
) -> None:
    """test-plan §21: Foundry Cohere malformed reply."""
    select()
    providers.script_foundry(ScriptedReply(shape=shape))

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as refused:
        _embed(operation)

    _assert_error_is_clean(refused.value)
    assert len(_foundry(providers)) == 1


@pytest.mark.parametrize("shape", ["short_count", "wrong_length"])
def test_a_malformed_second_batch_fails_the_whole_call_with_no_partial_result(
    select: Select, providers: FakeEmbeddingProviders, shape: str
) -> None:
    """test-plan §21: Foundry Cohere malformed reply (nothing is returned, so nothing can be written)."""
    select()
    providers.script_foundry(ScriptedReply(), ScriptedReply(shape=shape))

    with pytest.raises(embeddings_api.EmbeddingUnavailableError):
        embeddings_api.get_embeddings().embed_documents(_texts(100))


# ---- pacing --------------------------------------------------------------------------------


def test_the_pacing_interval_is_18_requests_per_minute() -> None:
    """test-plan §21: Foundry Cohere pacing (the constant)."""
    assert embeddings_api.FOUNDRY_COHERE_MIN_INTERVAL_SECONDS == pytest.approx(60 / 18)


def test_requests_of_every_kind_and_every_adapter_start_at_least_60_over_18_seconds_apart(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere pacing (queries and documents alike, process-wide)."""
    select()
    first = embeddings_api.get_embeddings()
    second = embeddings_api.get_embeddings()

    first.embed_documents(["a"])
    second.embed_query("b")
    first.embed_query("c")
    embeddings_api.embed_texts(["d"])
    embeddings_api.embed_query("e")
    second.embed_documents(["f"])

    requests = _foundry(providers)
    assert len(requests) == 6
    assert all(gap >= INTERVAL - TOLERANCE for gap in _gaps(requests))


def test_the_batches_of_one_large_call_are_paced_too(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere pacing (batches)."""
    select()

    embeddings_api.get_embeddings().embed_documents(_texts(200))

    requests = _foundry(providers)
    assert len(requests) == 3
    assert all(gap >= INTERVAL - TOLERANCE for gap in _gaps(requests))


def test_an_idle_process_is_not_held_back(
    select: Select, providers: FakeEmbeddingProviders, clock: FakeClock
) -> None:
    """test-plan §21: Foundry Cohere pacing (spacing is a minimum between starts, not a delay added to every call)."""
    select()
    embeddings = embeddings_api.get_embeddings()

    for n in range(3):
        embeddings.embed_query(f"question {n}")
        clock.advance(INTERVAL + 1)

    assert len(_foundry(providers)) == 3
    assert clock.slept == 0


# ---- throttle ------------------------------------------------------------------------------

RETRIED = [
    pytest.param(ScriptedReply(429, "7"), 7.0, id="429-retry-after-7"),
    pytest.param(ScriptedReply(429, "4.5"), 4.5, id="429-retry-after-fractional"),
    pytest.param(ScriptedReply(429, "1"), INTERVAL, id="429-short-retry-after-is-still-paced"),
    pytest.param(ScriptedReply(429), INTERVAL, id="429-no-retry-after"),
    pytest.param(ScriptedReply(429, "soon"), INTERVAL, id="429-invalid-retry-after"),
    pytest.param(ScriptedReply(429, "-5"), INTERVAL, id="429-negative-retry-after"),
    pytest.param(ScriptedReply(429, "600"), 60.0, id="429-retry-after-capped-at-60"),
    pytest.param(ScriptedReply(500), INTERVAL, id="500"),
    pytest.param(ScriptedReply(502, "9"), 9.0, id="502-retry-after-9"),
    pytest.param(ScriptedReply(503), INTERVAL, id="503"),
    pytest.param(ScriptedReply(drop=True), INTERVAL, id="connection-dropped"),
]


@OPERATIONS
@pytest.mark.parametrize(("failure", "waits"), RETRIED)
def test_a_throttled_or_failed_request_is_retried_once_after_the_retry_after_interval(
    select: Select,
    providers: FakeEmbeddingProviders,
    operation: str,
    failure: ScriptedReply,
    waits: float,
) -> None:
    """test-plan §21: Foundry Cohere throttle (wait min(Retry-After, 60), paced; retry once)."""
    select()
    providers.script_foundry(failure)

    vectors = _embed(operation)

    first, second = _foundry(providers)
    assert [len(vector) for vector in vectors] == [1024] * len(vectors)
    assert second.body == first.body
    waited = second.arrived_at - first.arrived_at
    assert waits - TOLERANCE <= waited <= waits + INTERVAL + TOLERANCE


@OPERATIONS
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_a_second_failure_raises_the_typed_error_naming_the_status_with_no_third_request(
    select: Select, providers: FakeEmbeddingProviders, operation: str, status: int
) -> None:
    """test-plan §21: Foundry Cohere throttle (retry also fails; error carries no key or body)."""
    select()
    providers.script_foundry(ScriptedReply(status, "2"), ScriptedReply(status, "2"))

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as failed:
        _embed(operation)

    assert str(status) in str(failed.value)
    _assert_error_is_clean(failed.value)
    assert len(_foundry(providers)) == 2


@OPERATIONS
@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_client_error_other_than_429_raises_at_once_without_a_retry(
    select: Select,
    providers: FakeEmbeddingProviders,
    clock: FakeClock,
    operation: str,
    status: int,
) -> None:
    """test-plan §21: Foundry Cohere throttle (a 4xx other than 429 raises at once)."""
    select()
    providers.script_foundry(ScriptedReply(status))

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as failed:
        _embed(operation)

    assert str(status) in str(failed.value)
    _assert_error_is_clean(failed.value)
    assert len(_foundry(providers)) == 1
    assert clock.slept == 0


@OPERATIONS
def test_an_unreachable_endpoint_is_retried_once_then_reported_as_unreachable(
    select: Select, providers: FakeEmbeddingProviders, clock: FakeClock, operation: str
) -> None:
    """test-plan §21: Foundry Cohere throttle (transport error: wait, retry once, then fail)."""
    select(AZURE_OPENAI_ENDPOINT=NO_ENDPOINT)

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as failed:
        _embed(operation)

    assert "unreachable" in str(failed.value).lower()
    _assert_error_is_clean(failed.value)
    assert clock.slept >= INTERVAL - TOLERANCE
    assert _foundry(providers) == []


def test_a_connection_dropped_twice_is_reported_as_unreachable_after_two_requests(
    select: Select, providers: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere throttle (transport error on the retry too)."""
    select()
    providers.script_foundry(ScriptedReply(drop=True), ScriptedReply(drop=True))

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as failed:
        _embed("query")

    assert "unreachable" in str(failed.value).lower()
    _assert_error_is_clean(failed.value)
    assert len(_foundry(providers)) == 2


# ---- missing configuration -----------------------------------------------------------------


@OPERATIONS
@pytest.mark.parametrize("setting", ["AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY"])
@pytest.mark.parametrize("how", ["unset", "blank"])
def test_a_missing_endpoint_or_key_is_named_before_any_request(
    select: Select, providers: FakeEmbeddingProviders, operation: str, setting: str, how: str
) -> None:
    """test-plan §21: Unknown or incomplete configuration (foundry_cohere loads without credentials;
    the first embedding call names the missing setting and sends nothing)."""
    if how == "unset":
        select(remove=(setting,))
    else:
        select(**{setting: ""})

    with pytest.raises(embeddings_api.EmbeddingUnavailableError) as failed:
        _embed(operation)

    assert setting in str(failed.value)
    _assert_error_is_clean(failed.value)
    assert providers.requests == []


# ---- run level -----------------------------------------------------------------------------


class _EmbeddingStore:
    """A filings store whose search embeds the query through the retrieval boundary, as a real one does."""

    def count_documents(self, filters: Any = None) -> int:
        return 3

    def search_by_ticker(
        self, query: str, ticker: str, n_results: int = 3, section: str | None = None
    ) -> Any:
        embeddings_api.embed_query(query)
        return SimpleNamespace(chunks=[])


@pytest.mark.asyncio
async def test_a_throttled_embedding_arm_records_a_recoverable_filing_retrieval_failure(
    monkeypatch: pytest.MonkeyPatch, foundry_embed: FakeEmbeddingProviders
) -> None:
    """test-plan §21: Foundry Cohere throttle (the filings step records a recoverable
    `filing_retrieval_failed`; other evidence is untouched; no key or body in the record)."""
    assert embeddings_api.embedding_identifier() == FOUNDRY_COHERE.identifier
    foundry_embed.script_foundry(ScriptedReply(429, "1"), ScriptedReply(429, "1"))
    monkeypatch.setattr(graph, "get_vector_store", lambda: _EmbeddingStore())
    state = create_initial_state("AAPL", "Apple Inc.")

    update = await graph.retrieve_sec_filings_node(state)

    assert update["filing_chunks"] == []
    (error,) = update["errors"]
    assert error["code"] == "filing_retrieval_failed"
    assert error["step"] == "retrieve_filings"
    assert error["recoverable"] is True
    assert error["message"] == "Filing retrieval failed for AAPL."
    assert len(foundry_embed.requests_to("foundry_cohere")) == 2
    assert AZURE_API_KEY not in str(update)
    assert LEAKY_BODY_MARKER not in str(update)
