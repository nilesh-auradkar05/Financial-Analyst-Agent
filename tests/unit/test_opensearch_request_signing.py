"""Request signing for the OpenSearch store (CM-3).

Every case runs the store against a local HTTP stand-in that records the headers of each
request it receives, with dummy AWS credentials and no network beyond 127.0.0.1.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from tests.rag.helpers import FakeEmbedder

INDEX_NAME = "cm3_signing_test_index"
DUMMY_KEY_ID = "AKIADUMMYKEYID123456"
DUMMY_SECRET = "dummy-secret-Zq8vK2mXp4Lr7TnYw3Hc9Bd5"
DUMMY_SESSION_TOKEN = "dummy-session-token-Jf6sGa1Uo0Ve8Ni2Rk4Pt7Mx"

_OPENSEARCH_ROOT: dict[str, Any] = {
    "version": {"number": "2.19.2", "distribution": "opensearch"},
    "tagline": "The OpenSearch Project: https://opensearch.org/",
}


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]


@dataclass
class StandIn:
    url: str
    requests: list[RecordedRequest] = field(default_factory=list)


def _make_handler(recorded: list[RecordedRequest]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            recorded.append(
                RecordedRequest(
                    method=self.command,
                    path=self.path,
                    headers={name.lower(): value for name, value in self.headers.items()},
                )
            )
            route = self.path.split("?", 1)[0]
            if route == "/" and self.command in ("GET", "HEAD"):
                status, body = 200, _OPENSEARCH_ROOT
            elif route.endswith("/_count"):
                status, body = 200, {"count": 0}
            else:
                status, body = 404, {}
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            self._respond()

        def do_POST(self) -> None:  # noqa: N802
            self._respond()

        def do_HEAD(self) -> None:  # noqa: N802
            self._respond()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


@pytest.fixture
def stand_in() -> Iterator[StandIn]:
    recorded: list[RecordedRequest] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(recorded))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield StandIn(url=f"http://127.0.0.1:{server.server_address[1]}", requests=recorded)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """Dummy static credentials only: no profile, container, web-identity or metadata source."""
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "OPENSEARCH_AUTH",
        "OPENSEARCH_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-such-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", DUMMY_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", DUMMY_SECRET)
    return monkeypatch


def _build(stand_in: StandIn) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(
        index_name=INDEX_NAME,
        url=stand_in.url,
        timeout_s=5.0,
        embeddings=FakeEmbedder(),
    )


_OPERATIONS: dict[str, Callable[[OpenSearchVectorStore], object]] = {
    "healthcheck": lambda store: store.healthcheck(),
    "count_documents": lambda store: store.count_documents(),
}


@pytest.mark.parametrize("operation", sorted(_OPERATIONS))
@pytest.mark.parametrize(
    ("region_env", "expected_region"),
    [(None, "us-east-1"), ("us-west-2", "us-west-2")],
    ids=["default-region", "explicit-region"],
)
def test_sigv4_requests_are_signed_for_aoss_in_region(
    aws_env: pytest.MonkeyPatch,
    stand_in: StandIn,
    operation: str,
    region_env: str | None,
    expected_region: str,
) -> None:
    """test-plan §5: Request signing (CM-3) - sigv4 signs every request for aoss in AWS_REGION."""
    aws_env.setenv("OPENSEARCH_AUTH", "sigv4")
    if region_env is not None:
        aws_env.setenv("AWS_REGION", region_env)

    _OPERATIONS[operation](_build(stand_in))

    assert stand_in.requests, "the store sent no request to the stand-in"
    for request in stand_in.requests:
        authorization = request.headers.get("authorization", "")
        assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={DUMMY_KEY_ID}/"), (
            f"{request.method} {request.path} is not SigV4-signed"
        )
        scope = re.search(r"Credential=[^,\s]+", authorization)
        assert scope is not None
        assert scope.group(0).endswith(f"/{expected_region}/aoss/aws4_request")
        assert request.headers.get("x-amz-content-sha256")


@pytest.mark.parametrize("operation", sorted(_OPERATIONS))
@pytest.mark.parametrize("auth_value", [None, "none"], ids=["unset", "none"])
def test_unset_or_none_auth_sends_unsigned_requests(
    aws_env: pytest.MonkeyPatch,
    stand_in: StandIn,
    operation: str,
    auth_value: str | None,
) -> None:
    """test-plan §5: Request signing (CM-3) - unset or `none` leaves requests unsigned."""
    if auth_value is not None:
        aws_env.setenv("OPENSEARCH_AUTH", auth_value)

    _OPERATIONS[operation](_build(stand_in))

    assert stand_in.requests, "the store sent no request to the stand-in"
    for request in stand_in.requests:
        assert "authorization" not in request.headers
        assert "x-amz-content-sha256" not in request.headers


@pytest.mark.parametrize("auth_value", ["basic", "SIGV4X"])
def test_invalid_auth_value_fails_at_construction(
    aws_env: pytest.MonkeyPatch, stand_in: StandIn, auth_value: str
) -> None:
    """test-plan §5: Request signing (CM-3) - any other OPENSEARCH_AUTH value errors before a request."""
    aws_env.setenv("OPENSEARCH_AUTH", auth_value)

    with pytest.raises(ValueError):
        _build(stand_in)

    assert stand_in.requests == []


def test_sigv4_without_credentials_fails_at_construction(
    aws_env: pytest.MonkeyPatch, stand_in: StandIn
) -> None:
    """test-plan §5: Request signing (CM-3) - sigv4 with no resolvable credentials errors before a request."""
    aws_env.setenv("OPENSEARCH_AUTH", "sigv4")
    aws_env.delenv("AWS_ACCESS_KEY_ID")
    aws_env.delenv("AWS_SECRET_ACCESS_KEY")

    with pytest.raises((ValueError, RuntimeError)):
        _build(stand_in)

    assert stand_in.requests == []


def test_sigv4_credentials_never_appear_in_stats_or_logs(
    aws_env: pytest.MonkeyPatch,
    stand_in: StandIn,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """test-plan §5: Request signing (CM-3) - no credential appears in get_stats() or logs."""
    aws_env.setenv("OPENSEARCH_AUTH", "sigv4")
    aws_env.setenv("AWS_SESSION_TOKEN", DUMMY_SESSION_TOKEN)
    loguru_lines: list[str] = []
    handler_id = logger.add(loguru_lines.append, level="DEBUG")
    caplog.set_level("DEBUG")
    try:
        store = _build(stand_in)
        stats = store.get_stats()
        store.healthcheck()
    finally:
        logger.remove(handler_id)

    assert stand_in.requests, "the store sent no request to the stand-in"
    assert all(
        request.headers.get("authorization", "").startswith("AWS4-HMAC-SHA256 ")
        for request in stand_in.requests
    ), "requests were not signed, so the leak check would prove nothing"
    output = repr(stats) + "".join(loguru_lines) + caplog.text
    assert DUMMY_SECRET not in output
    assert DUMMY_SESSION_TOKEN not in output
