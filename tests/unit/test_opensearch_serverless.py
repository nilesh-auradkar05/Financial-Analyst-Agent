"""OpenSearch Serverless behavior of the OpenSearch store (test-plan §5, rows marked CM-3).

Every case runs the store against a local HTTP stand-in that answers with the status codes and
errors captured from the NEXTGEN collection on 2026-11-08, with dummy AWS credentials and no
network beyond 127.0.0.1:

- `HEAD /` and `GET /` answer 404; `HEAD /<index>` answers 200 when the index exists, 404 when not.
- `_bulk?refresh=wait_for` is rejected with 400; a bulk delete of an id that is already gone
  answers 200 overall with `errors: true` and a `not_found` item.
- Search is eventually consistent: deleted ids keep appearing in results ("stale" mode).
- Index creation naming `engine` or `mode` is rejected with 400.

With `serverless=False` the stand-in behaves like the local container instead: `/` answers 200 and
nothing is rejected.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from tests.embedding_arms import use_embedding_settings
from tests.rag.helpers import FakeEmbedder, build_corpus

INDEX_NAME = "cm3_serverless_test_index"
DUMMY_KEY_ID = "AKIADUMMYKEYID123456"
DUMMY_SECRET = "dummy-secret-Zq8vK2mXp4Lr7TnYw3Hc9Bd5"

_OPENSEARCH_ROOT: dict[str, Any] = {
    "version": {"number": "2.19.2", "distribution": "opensearch"},
    "tagline": "The OpenSearch Project: https://opensearch.org/",
}
_MAX_SEARCHES = 400  # a store that never makes progress ends in an error, not an endless loop


@dataclass(frozen=True)
class Recorded:
    method: str
    path: str
    query: dict[str, list[str]]
    body: bytes

    def as_json(self) -> Any:
        return json.loads(self.body)

    def ndjson(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.body.decode("utf-8").splitlines() if line.strip()]


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _term_value(value: Any, field_name: str) -> str | None:
    """The value of the first `{"term": {field_name: value}}` clause found anywhere in a query."""
    if isinstance(value, dict):
        term = value.get("term")
        if isinstance(term, dict) and field_name in term:
            return str(term[field_name])
        for item in value.values():
            found = _term_value(item, field_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _term_value(item, field_name)
            if found is not None:
                return found
    return None


def embedding_mapping(level: str) -> dict[str, Any]:
    """The `properties` of an index the store created at compression `level`, as the backend reports it."""
    return {
        "embedding": {
            "type": "knn_vector",
            "dimension": FakeEmbedder.dimension,
            "compression_level": level,
            "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
        }
    }


class StandIn:
    """A configurable OpenSearch / OpenSearch Serverless stand-in listening on 127.0.0.1."""

    def __init__(
        self,
        *,
        serverless: bool = True,
        index_exists: bool = True,
        compression_level: str = "32x",
        head_index_status: int | None = None,
        delay_s: float = 0.0,
        drop_first: int = 0,
        documents: dict[str, str] | None = None,
    ) -> None:
        self.serverless = serverless
        self.head_index_status = head_index_status
        self.delay_s = delay_s
        self.requests: list[Recorded] = []
        self.created_bodies: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._drops_left = drop_first
        self._index_exists = index_exists
        self._properties: dict[str, Any] | None = (
            embedding_mapping(compression_level) if index_exists else None
        )
        self._documents: dict[str, str] = dict(documents or {})  # chunk id -> ticker
        self._deleted: set[str] = set()
        self._searches = 0
        self._server = _Server(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def bulk_requests(self) -> list[Recorded]:
        return [request for request in self.requests if request.path.endswith("/_bulk")]

    def deleted_ids(self, since: int = 0) -> list[str]:
        """Ids named by `delete` lines in the bulk requests received after the first `since` requests."""
        return [
            str(action["delete"]["_id"])
            for request in self.requests[since:]
            if request.path.endswith("/_bulk")
            for action in request.ndjson()
            if "delete" in action
        ]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                target = urlsplit(self.path)
                request = Recorded(self.command, target.path, parse_qs(target.query), body)
                with stand_in._lock:
                    stand_in.requests.append(request)
                    drop = stand_in._drops_left > 0
                    if drop:
                        stand_in._drops_left -= 1
                if drop:
                    self.close_connection = True  # request read, socket closed, no response
                    return
                if stand_in.delay_s:
                    time.sleep(stand_in.delay_s)
                status, payload = stand_in._answer(request)
                data = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(data)
                except OSError:
                    self.close_connection = True

            do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = _serve  # noqa: N815

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return Handler

    def _answer(self, request: Recorded) -> tuple[int, Any]:
        path, method = request.path, request.method
        index_path = f"/{INDEX_NAME}"
        with self._lock:
            if path == "/" and method in ("GET", "HEAD"):
                return (404, {}) if self.serverless else (200, _OPENSEARCH_ROOT)
            if path == index_path and method == "HEAD":
                if self.head_index_status is not None:
                    return self.head_index_status, {}
                return (200 if self._index_exists else 404), {}
            if path == index_path and method == "PUT":
                return self._create(request)
            if path == f"{index_path}/_mapping" and method == "GET":
                if self._properties is None:
                    return 404, {"error": "index_not_found_exception", "status": 404}
                return 200, {INDEX_NAME: {"mappings": {"properties": self._properties}}}
            if path in ("/_bulk", f"{index_path}/_bulk") and method == "POST":
                return self._bulk(request)
            if path == f"{index_path}/_search" and method == "POST":
                return self._search(request)
            if path == f"{index_path}/_count" and method == "POST":
                return 200, {"count": len(self._documents)}
            if path == f"{index_path}/_count" and method == "GET":
                return 200, {"count": len(self._documents)}
        return 404, {"error": "no handler", "status": 404}

    def _create(self, request: Recorded) -> tuple[int, Any]:
        body = request.as_json()
        self.created_bodies.append(body)
        if self.serverless:
            for rejected in ("engine", "mode"):
                if _contains_key(body, rejected):
                    return 400, {
                        "error": {
                            "type": "mapper_parsing_exception",
                            "reason": f"Field parameter '{rejected}' is not supported",
                        },
                        "status": 400,
                    }
        self._index_exists = True
        self._properties = body.get("mappings", {}).get("properties", {})
        return 200, {"acknowledged": True, "shards_acknowledged": True, "index": INDEX_NAME}

    def _bulk(self, request: Recorded) -> tuple[int, Any]:
        if self.serverless and "wait_for" in request.query.get("refresh", []):
            return 400, {
                "error": {
                    "type": "illegal_argument_exception",
                    "reason": "wait_for refresh policy is not supported",
                },
                "status": 400,
            }
        lines = request.ndjson()
        items: list[dict[str, Any]] = []
        position = 0
        while position < len(lines):
            action = lines[position]
            operation, meta = next(iter(action.items()))
            document_id = str(meta["_id"])
            position += 1
            if operation == "delete":
                if document_id in self._deleted or document_id not in self._documents:
                    items.append(
                        {
                            "delete": {
                                "_index": INDEX_NAME,
                                "_id": document_id,
                                "result": "not_found",
                                "status": 404,
                            }
                        }
                    )
                else:
                    self._deleted.add(document_id)  # searches keep returning it (stale)
                    items.append(
                        {
                            "delete": {
                                "_index": INDEX_NAME,
                                "_id": document_id,
                                "result": "deleted",
                                "status": 200,
                            }
                        }
                    )
                continue
            source = lines[position]
            position += 1
            self._documents[document_id] = str(source.get("ticker", ""))
            self._deleted.discard(document_id)
            items.append(
                {
                    operation: {
                        "_index": INDEX_NAME,
                        "_id": document_id,
                        "result": "created",
                        "status": 201,
                    }
                }
            )
        errors = any(next(iter(item.values()))["status"] >= 400 for item in items)
        return 200, {"took": 1, "errors": errors, "items": items}

    def _search(self, request: Recorded) -> tuple[int, Any]:
        self._searches += 1
        if self._searches > _MAX_SEARCHES:
            return 500, {"error": "stand-in search limit reached", "status": 500}
        body = request.as_json()
        ticker = _term_value(body.get("query", {}), "ticker")
        ids = sorted(
            document_id
            for document_id, owner in self._documents.items()
            if ticker is None or owner == ticker
        )
        after = body.get("search_after")
        if after:
            ids = [document_id for document_id in ids if document_id > str(after[0])]
        page = ids[: int(body.get("size", 10))]
        hits: list[dict[str, Any]] = []
        for document_id in page:
            hit: dict[str, Any] = {"_index": INDEX_NAME, "_id": document_id, "_score": None}
            if body.get("_source") is not False:
                hit["_source"] = {"ticker": self._documents[document_id]}
            if "sort" in body:
                hit["sort"] = [document_id]
            hits.append(hit)
        return 200, {
            "took": 1,
            "timed_out": False,
            "hits": {"total": {"value": len(ids), "relation": "eq"}, "hits": hits},
        }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        return


@pytest.fixture
def start_stand_in() -> Iterator[Callable[..., StandIn]]:
    started: list[StandIn] = []

    def start(**config: Any) -> StandIn:
        stand_in = StandIn(**config)
        started.append(stand_in)
        return stand_in

    yield start
    for stand_in in started:
        stand_in.close()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """16-d fake embedding settings, dummy static AWS credentials and no OpenSearch variable set."""
    use_embedding_settings(
        monkeypatch,
        tmp_path,
        {
            "EMBEDDING_PROVIDER": "bedrock",
            "EMBEDDING_MODEL": FakeEmbedder.model,
            "EMBEDDING_DIMENSIONS": str(FakeEmbedder.dimension),
        },
    )
    for name in (
        "OPENSEARCH_AUTH",
        "OPENSEARCH_URL",
        "OPENSEARCH_INDEX_NAME",
        "OPENSEARCH_TIMEOUT_S",
        "OPENSEARCH_COMPRESSION_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", DUMMY_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", DUMMY_SECRET)
    return monkeypatch


def _build(stand_in: StandIn, timeout_s: float | None = None) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(
        index_name=INDEX_NAME,
        url=stand_in.url,
        timeout_s=timeout_s,
        embeddings=FakeEmbedder(),
    )


def _auth(env: pytest.MonkeyPatch, mode: str) -> bool:
    """Select the OPENSEARCH_AUTH mode; returns whether the backend to stand in for is Serverless."""
    env.setenv("OPENSEARCH_AUTH", mode)
    return mode == "sigv4"


# --- Health check (test-plan §5: Health check (CM-3)) -------------------------------------------


@pytest.mark.parametrize("auth", ["none", "sigv4"])
@pytest.mark.parametrize(
    ("status", "available"),
    [(200, True), (404, True), (401, False), (403, False), (500, False)],
)
def test_healthcheck_probes_the_index_and_reads_its_status(
    env: pytest.MonkeyPatch,
    start_stand_in: Callable[..., StandIn],
    auth: str,
    status: int,
    available: bool,
) -> None:
    """test-plan §5: Health check (CM-3) - HEAD /<index>: 200 or 404 available; 401, 403 and any
    other error status unavailable (Serverless answers `/` with 404)."""
    stand_in = start_stand_in(serverless=_auth(env, auth), head_index_status=status)

    healthy = _build(stand_in).healthcheck()

    assert {(r.method, r.path) for r in stand_in.requests} == {("HEAD", f"/{INDEX_NAME}")}
    assert healthy is available


@pytest.mark.parametrize("auth", ["none", "sigv4"])
def test_healthcheck_reports_connection_error_as_unavailable(
    env: pytest.MonkeyPatch, auth: str
) -> None:
    """test-plan §5: Health check (CM-3) - a connection error is unavailable, promptly."""
    env.setenv("OPENSEARCH_AUTH", auth)
    store = OpenSearchVectorStore(
        index_name=INDEX_NAME, url="http://127.0.0.1:1", timeout_s=1, embeddings=FakeEmbedder()
    )

    started = time.monotonic()
    healthy = store.healthcheck()
    elapsed = time.monotonic() - started

    assert healthy is False
    assert elapsed < 3


def test_healthcheck_reports_a_timeout_as_unavailable_within_the_timeout(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Health check (CM-3) - a timeout is unavailable; an explicit timeout_s is honored."""
    stand_in = start_stand_in(serverless=_auth(env, "sigv4"), delay_s=3.0)

    started = time.monotonic()
    healthy = _build(stand_in, timeout_s=1.0).healthcheck()
    elapsed = time.monotonic() - started

    assert {r.path for r in stand_in.requests} == {f"/{INDEX_NAME}"}, "the index was never probed"
    assert healthy is False
    assert elapsed < 2.5


# --- Serverless timeout and retry (test-plan §5: Serverless timeout and retry (CM-3)) ------------


def test_sigv4_default_timeout_outlasts_a_cold_start(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Serverless timeout and retry (CM-3) - sigv4 with no OPENSEARCH_TIMEOUT_S waits
    30 s, so a 6 s cold start still answers."""
    stand_in = start_stand_in(serverless=_auth(env, "sigv4"), delay_s=6.0, head_index_status=200)

    assert _build(stand_in).healthcheck() is True


def test_unsigned_default_timeout_is_five_seconds(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Serverless timeout and retry (CM-3) - unset or `none` keeps the 5 s timeout."""
    stand_in = start_stand_in(serverless=_auth(env, "none"), delay_s=6.0, head_index_status=200)

    assert _build(stand_in).healthcheck() is False


@pytest.mark.parametrize("auth", ["none", "sigv4"])
def test_explicit_timeout_variable_applies_in_both_modes(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn], auth: str
) -> None:
    """test-plan §5: Serverless timeout and retry (CM-3) - OPENSEARCH_TIMEOUT_S applies in both
    modes (here it is shorter than either default, so a 2.5 s answer is too late)."""
    stand_in = start_stand_in(serverless=_auth(env, auth), delay_s=2.5, head_index_status=200)
    env.setenv("OPENSEARCH_TIMEOUT_S", "1")

    assert _build(stand_in).healthcheck() is False


def test_constructor_timeout_takes_precedence_over_the_timeout_variable(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Serverless timeout and retry (CM-3) - timeout_s > OPENSEARCH_TIMEOUT_S."""
    stand_in = start_stand_in(serverless=_auth(env, "sigv4"), delay_s=2.5, head_index_status=200)
    env.setenv("OPENSEARCH_TIMEOUT_S", "30")

    assert _build(stand_in, timeout_s=1.0).healthcheck() is False


@pytest.mark.parametrize(
    ("auth", "drops", "available", "attempts"),
    [
        ("sigv4", 1, True, 2),
        ("sigv4", 2, False, 2),
        ("none", 1, False, 1),
    ],
    ids=["sigv4-one-drop-retried", "sigv4-two-drops-error", "none-no-retry"],
)
def test_dropped_connection_is_retried_once_only_under_sigv4(
    env: pytest.MonkeyPatch,
    start_stand_in: Callable[..., StandIn],
    auth: str,
    drops: int,
    available: bool,
    attempts: int,
) -> None:
    """test-plan §5: Serverless timeout and retry (CM-3) - under sigv4 a dropped connection is
    retried once and a second drop is an error; unset or `none` makes exactly one attempt."""
    stand_in = start_stand_in(serverless=_auth(env, auth), drop_first=drops)

    healthy = _build(stand_in).healthcheck()

    assert len(stand_in.requests) == attempts
    assert healthy is available


# --- Serverless writes (test-plan §5: Serverless writes (CM-3)) ---------------------------------


@pytest.mark.parametrize(
    ("auth", "refresh"),
    [("sigv4", None), ("none", ["wait_for"])],
    ids=["sigv4-no-refresh", "none-wait-for"],
)
def test_add_documents_sends_refresh_only_in_local_mode(
    env: pytest.MonkeyPatch,
    start_stand_in: Callable[..., StandIn],
    auth: str,
    refresh: list[str] | None,
) -> None:
    """test-plan §5: Serverless writes (CM-3) - under sigv4 `_bulk` carries no `refresh` parameter;
    unset or `none` keeps refresh=wait_for."""
    stand_in = start_stand_in(serverless=_auth(env, auth))
    documents = build_corpus()[:3]

    written = _build(stand_in).add_documents(documents)

    assert written == len(documents)
    bulks = stand_in.bulk_requests()
    assert bulks, "add_documents sent no _bulk request"
    for bulk in bulks:
        assert bulk.query.get("refresh") == refresh


def _bulk_stand_in(env: pytest.MonkeyPatch, start: Callable[..., StandIn]) -> StandIn:
    documents = {f"AAPL_{number:05d}": "AAPL" for number in range(1201)}
    documents.update({f"MSFT_{number:05d}": "MSFT" for number in range(40)})
    return start(serverless=_auth(env, "sigv4"), documents=documents)


def test_delete_by_ticker_pages_past_the_page_size_and_deletes_each_id_once(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Delete by ticker (CM-3) - with 1,201 matching ids on stale search, every id is
    collected across the 500-hit pages, deleted exactly once, and counted; other tickers survive;
    Serverless writes (CM-3) - no `refresh` parameter on any `_bulk`."""
    stand_in = _bulk_stand_in(env, start_stand_in)

    deleted = _build(stand_in).delete_by_ticker("AAPL")

    expected = {f"AAPL_{number:05d}" for number in range(1201)}
    sent = Counter(stand_in.deleted_ids())
    assert deleted == 1201
    assert set(sent) == expected
    assert max(sent.values()) == 1
    for bulk in stand_in.bulk_requests():
        assert "refresh" not in bulk.query


def test_delete_by_ticker_tolerates_ids_already_gone(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn]
) -> None:
    """test-plan §5: Delete by ticker (CM-3) - a delete answered `not_found` is not an error, so a
    second call made before the first call's deletes are visible neither fails nor repeats a delete."""
    stand_in = _bulk_stand_in(env, start_stand_in)
    store = _build(stand_in)
    store.delete_by_ticker("AAPL")
    already_sent = len(stand_in.requests)

    deleted_again = store.delete_by_ticker("AAPL")

    resent = Counter(stand_in.deleted_ids(since=already_sent))
    assert deleted_again == 1201
    assert set(resent) == {f"AAPL_{number:05d}" for number in range(1201)}
    assert max(resent.values()) == 1


# --- Vector precision (test-plan §5: Vector precision (CM-3)) -----------------------------------


@pytest.mark.parametrize("level", ["2x", "32X", "full"])
def test_unknown_compression_level_fails_at_construction(
    env: pytest.MonkeyPatch, start_stand_in: Callable[..., StandIn], level: str
) -> None:
    """test-plan §5: Vector precision (CM-3) - any OPENSEARCH_COMPRESSION_LEVEL other than 1x or
    32x is an error at construction, before any request."""
    stand_in = start_stand_in()
    env.setenv("OPENSEARCH_COMPRESSION_LEVEL", level)

    with pytest.raises(ValueError):
        _build(stand_in)

    assert stand_in.requests == []


@pytest.mark.parametrize(
    ("configured", "expected_level"),
    [(None, "32x"), ("32x", "32x"), ("1x", "1x")],
    ids=["default", "32x", "1x"],
)
def test_created_vector_field_sets_compression_and_names_no_engine_or_mode(
    env: pytest.MonkeyPatch,
    start_stand_in: Callable[..., StandIn],
    configured: str | None,
    expected_level: str,
) -> None:
    """test-plan §5: Vector precision (CM-3) - the created vector field sets compression_level from
    OPENSEARCH_COMPRESSION_LEVEL (32x by default) and names no `engine` and no `mode`, which
    Serverless rejects."""
    stand_in = start_stand_in(serverless=_auth(env, "sigv4"), index_exists=False)
    if configured is not None:
        env.setenv("OPENSEARCH_COMPRESSION_LEVEL", configured)
    documents = build_corpus()[:3]

    written = _build(stand_in).add_documents(documents)

    assert written == len(documents)
    assert len(stand_in.created_bodies) == 1
    created = stand_in.created_bodies[0]
    vector = created["mappings"]["properties"]["embedding"]
    assert vector["type"] == "knn_vector"
    assert vector["dimension"] == FakeEmbedder.dimension
    assert vector["compression_level"] == expected_level
    assert vector["method"]["name"] == "hnsw"
    assert vector["method"]["space_type"] == "cosinesimil"
    assert not _contains_key(created, "engine")
    assert not _contains_key(created, "mode")
