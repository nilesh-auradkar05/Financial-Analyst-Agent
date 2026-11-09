"""A faithful fake of the embedding endpoints the four embedding arms call (test-plan §21).

One real HTTP server on 127.0.0.1 serves three wire formats, so the production clients build and
send real requests and this server parses them:

- Foundry Cohere `POST /providers/cohere/v2/embed` (deployment `cohere-embed-v5-fast`, up to 96
  texts per request, float embeddings, `output_dimension` among 256/512/1024/1536). Replies can be
  scripted to misbehave the way the 20 RPM deployment does (429 with or without `Retry-After`, 5xx,
  4xx, a dropped connection, a wrong vector count or length); unscripted requests get honest
  vectors. Every error body carries `LEAKY_BODY_MARKER`, so a test can show no error text echoes it.
  `enforce_quota` makes it enforce the deployment's rolling 60 s limits (requests and tokens, a
  text counting ceil(characters / 4) tokens), answering 429 with `Retry-After` when exceeded.
- Bedrock `POST /model/{modelId}/invoke` (InvokeModel; the model id arrives percent-encoded):
  Titan Text Embeddings V2 (`inputText`, `dimensions`, native 1024) and Cohere Embed v4 (`texts`,
  `input_type`, `output_dimension`, native 1536). A request carrying neither a SigV4 signature nor a bearer token is refused.
- Azure / Foundry `POST /openai/v1/embeddings` for one deployment (`embed-3-small`, native 1536),
  honouring `encoding_format: "base64"`. `input` must be text; token-id arrays are refused, as a
  deployment cannot be assumed to accept them.

Every vector is computed from the model, the text and the requested dimension (hash-seeded, unit
length); nothing is replayed from a test. A dimension the model does not support is refused with a
400, and a missing dimension field falls back to the model's native size, so a client that drops the
field is visible in the vector length. No network beyond loopback; credentials are kept apart from
the recorded headers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

FOUNDRY_MODEL = "cohere-embed-v5-fast"
FOUNDRY_PATH = "/providers/cohere/v2/embed"
FOUNDRY_DIMENSIONS = (256, 512, 1024, 1536)
FOUNDRY_MAX_TEXTS = 96
LEAKY_BODY_MARKER = "upstream-diagnostic-body-must-not-leak"
TITAN_MODEL = "amazon.titan-embed-text-v2:0"
COHERE_MODEL = "cohere.embed-v4:0"
AZURE_DEPLOYMENT = "embed-3-small"
AZURE_NATIVE_DIMENSION = 1536

AZURE_PATH = "/openai/v1/embeddings"
_BEDROCK_PATH = re.compile(r"^/model/(?P<model>[^/]+)/invoke$")
_SIGV4_CREDENTIAL = re.compile(r"^AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/")
_AUTH_HEADERS = ("api-key", "authorization")

TITAN_DIMENSIONS = (256, 512, 1024)
COHERE_DIMENSIONS = (256, 512, 1024, 1536)
COHERE_INPUT_TYPES = ("search_document", "search_query", "classification", "clustering")


def vector_for(model: str, text: str, dimension: int, input_type: str = "") -> list[float]:
    """The unit vector this fake returns for `text` from `model` at `dimension`."""
    seed = int.from_bytes(hashlib.sha256(f"{model}|{input_type}|{text}".encode()).digest(), "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


_REAL_MONOTONIC = time.monotonic
_clock_offset = 0.0  # only ever grows, so the fake clock is monotonic across tests
_clock_lock = threading.Lock()


class FakeClock:
    """The process clock with sleeping made free: `time.monotonic()` is the real monotonic clock
    plus every second slept or advanced so far; `time.sleep(s)` returns at once and adds `s`.

    Real time keeps flowing underneath, so asyncio's loop clock (which reads `time.monotonic`)
    still advances, and the offset never shrinks between tests. A new clock starts with the
    process idle for a while, so the first request of a test is never held back by the pacing
    of an earlier test.
    """

    IDLE_AT_START_SECONDS = 10.0

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.advance(self.IDLE_AT_START_SECONDS)

    def monotonic(self) -> float:
        return _REAL_MONOTONIC() + _clock_offset

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        global _clock_offset
        with _clock_lock:
            _clock_offset += seconds

    @property
    def slept(self) -> float:
        return sum(self.sleeps)


@dataclass
class ReceivedEmbeddingRequest:
    """One request as the provider saw it. `provider` is the route that served it."""

    provider: str
    path: str
    model: str
    headers: dict[str, str]
    credential: str
    body: dict[str, Any]
    scheme: str = ""  # "sigv4", "bearer", "api-key" or "" when the request carried no credential
    arrived_at: float = 0.0  # the fake clock when the request reached the server


@dataclass
class ScriptedReply:
    """What the Foundry route answers instead of an honest reply, once.

    `status` >= 400 answers that error (its body carries LEAKY_BODY_MARKER) with `retry_after`
    as the `Retry-After` header when given. `drop` closes the connection without answering.
    A 200 with `shape` answers real vectors that are wrong: "short_count" omits the last vector,
    "wrong_length" returns vectors one dimension too short.
    """

    status: int = 200
    retry_after: str | None = None
    drop: bool = False
    shape: str = ""


@dataclass
class _Reply:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    drop: bool = False


class _RefusedError(Exception):
    def __init__(
        self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(status)
        self.status = status
        self.payload = payload
        self.headers = headers or {}


def _bad_request(message: str) -> _RefusedError:
    return _RefusedError(400, {"error": {"code": "BadRequest", "message": message}})


def _texts(value: Any) -> list[str]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(t, str) for t in items):
        raise _bad_request("input must be a non-empty string or list of strings")
    return items


class FakeEmbeddingProviders:
    """Serves Foundry Cohere, Bedrock (Titan V2, Cohere v4) and Azure embedding replies computed from each request."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.requests: list[ReceivedEmbeddingRequest] = []
        self._clock: Callable[[], float] = clock.monotonic if clock else (lambda: 0.0)
        self._script: deque[ScriptedReply] = deque()
        self._quota: tuple[int, int] | None = None
        self._admitted: deque[tuple[float, int]] = deque()
        self.quota_rejections: list[ReceivedEmbeddingRequest] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeEmbeddingProviders:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def script_foundry(self, *replies: ScriptedReply) -> None:
        """Answer the next Foundry requests with these replies, in order; honest replies after."""
        with self._lock:
            self._script.extend(replies)

    def enforce_quota(self, *, requests_per_minute: int, tokens_per_minute: int) -> None:
        """Refuse Foundry requests that would exceed either limit within the last 60 s of the fake clock."""
        with self._lock:
            self._quota = (requests_per_minute, tokens_per_minute)

    def _quota_wait(self, now: float, tokens: int) -> int | None:
        """Seconds until a request of `tokens` fits, or None when it fits now (and is admitted)."""
        with self._lock:
            if self._quota is None:
                return None
            max_requests, max_tokens = self._quota
            window = [(at, used) for at, used in self._admitted if now - at < 60.0]
            self._admitted = deque(window)

            def fits(held: list[tuple[float, int]]) -> bool:
                return (
                    len(held) + 1 <= max_requests
                    and sum(used for _, used in held) + tokens <= max_tokens
                )

            if fits(window):
                self._admitted.append((now, tokens))
                return None
            wait = 60.0
            remaining = list(window)
            while remaining:
                freed_at = remaining.pop(0)[0] + 60.0 - now
                if fits(remaining):
                    wait = freed_at
                    break
            return max(1, math.ceil(wait))

    def requests_to(self, provider: str) -> list[ReceivedEmbeddingRequest]:
        return [request for request in self.requests if request.provider == provider]

    def _record(self, request: ReceivedEmbeddingRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def _next_scripted(self) -> ScriptedReply | None:
        with self._lock:
            return self._script.popleft() if self._script else None

    def _foundry(self, request: ReceivedEmbeddingRequest) -> _Reply:
        scripted = self._next_scripted()
        if scripted and scripted.drop:
            return _Reply(0, {}, drop=True)
        if scripted and scripted.status >= 400:
            headers = {"Retry-After": scripted.retry_after} if scripted.retry_after else {}
            raise _RefusedError(
                scripted.status,
                {"error": {"code": str(scripted.status), "message": LEAKY_BODY_MARKER}},
                headers,
            )
        body = request.body
        if request.credential == "":
            raise _RefusedError(401, {"error": {"code": "401", "message": "Access denied: no key"}})
        if body.get("model") != FOUNDRY_MODEL:
            raise _RefusedError(
                404, {"error": {"code": "DeploymentNotFound", "message": "no such deployment"}}
            )
        texts = _texts(body.get("texts"))
        if len(texts) > FOUNDRY_MAX_TEXTS:
            raise _bad_request(f"at most {FOUNDRY_MAX_TEXTS} texts per request")
        input_type = body.get("input_type")
        if input_type not in COHERE_INPUT_TYPES:
            raise _bad_request(f"input_type must be one of {COHERE_INPUT_TYPES}")
        dimension = body.get("output_dimension", FOUNDRY_DIMENSIONS[-1])
        if dimension not in FOUNDRY_DIMENSIONS:
            raise _bad_request(f"output_dimension must be one of {FOUNDRY_DIMENSIONS}")
        if body.get("embedding_types") != ["float"]:
            raise _bad_request("embedding_types must be ['float']")
        tokens = sum(math.ceil(len(text) / 4) for text in texts)
        wait = self._quota_wait(request.arrived_at, tokens)
        if wait is not None:
            with self._lock:
                self.quota_rejections.append(request)
            raise _RefusedError(
                429,
                {"error": {"code": "429", "message": LEAKY_BODY_MARKER}},
                {"Retry-After": str(wait)},
            )
        shape = scripted.shape if scripted else ""
        served = dimension - 1 if shape == "wrong_length" else dimension
        vectors = [vector_for(FOUNDRY_MODEL, text, served, input_type) for text in texts]
        if shape == "short_count":
            vectors = vectors[:-1]
        return _Reply(
            200,
            {
                "id": "embed_fake",
                "response_type": "embeddings_by_type",
                "embeddings": {"float": vectors},
                "texts": texts,
            },
        )

    def _bedrock(self, request: ReceivedEmbeddingRequest) -> dict[str, Any]:
        if request.model == TITAN_MODEL:
            return self._titan(request.body)
        if request.model == COHERE_MODEL:
            return self._cohere(request.body)
        raise _RefusedError(
            404, {"message": f"Could not resolve the foundation model {request.model!r}"}
        )

    def _titan(self, body: dict[str, Any]) -> dict[str, Any]:
        text = body.get("inputText")
        if not isinstance(text, str) or not text:
            raise _bad_request("inputText must be a non-empty string")
        dimension = body.get("dimensions", TITAN_DIMENSIONS[-1])
        if dimension not in TITAN_DIMENSIONS:
            raise _bad_request(f"dimensions must be one of {TITAN_DIMENSIONS}")
        return {
            "embedding": vector_for(TITAN_MODEL, text, dimension),
            "inputTextTokenCount": len(text.split()),
        }

    def _cohere(self, body: dict[str, Any]) -> dict[str, Any]:
        texts = _texts(body.get("texts"))
        input_type = body.get("input_type")
        if input_type not in COHERE_INPUT_TYPES:
            raise _bad_request(f"input_type must be one of {COHERE_INPUT_TYPES}")
        dimension = body.get("output_dimension", COHERE_DIMENSIONS[-1])
        if dimension not in COHERE_DIMENSIONS:
            raise _bad_request(f"output_dimension must be one of {COHERE_DIMENSIONS}")
        if "float" not in body.get("embedding_types", ["float"]):
            raise _bad_request("only float embeddings are served")
        return {
            "id": "embed_fake",
            "response_type": "embeddings_by_type",
            "embeddings": {
                "float": [vector_for(COHERE_MODEL, t, dimension, input_type) for t in texts]
            },
            "texts": texts,
        }

    def _azure(self, request: ReceivedEmbeddingRequest) -> dict[str, Any]:
        body = request.body
        if request.credential == "":
            raise _RefusedError(401, {"error": {"code": "401", "message": "Access denied: no key"}})
        if body.get("model") != AZURE_DEPLOYMENT:
            raise _RefusedError(
                404, {"error": {"code": "DeploymentNotFound", "message": "no such deployment"}}
            )
        texts = _texts(body.get("input"))
        dimension = body.get("dimensions", AZURE_NATIVE_DIMENSION)
        if not isinstance(dimension, int) or not 1 <= dimension <= AZURE_NATIVE_DIMENSION:
            raise _bad_request(f"dimensions must be between 1 and {AZURE_NATIVE_DIMENSION}")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise _bad_request("encoding_format must be float or base64")
        data = []
        for index, text in enumerate(texts):
            vector = vector_for(AZURE_DEPLOYMENT, text, dimension)
            payload: Any = vector
            if encoding == "base64":
                payload = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
            data.append({"object": "embedding", "index": index, "embedding": payload})
        tokens = sum(len(text.split()) for text in texts)
        return {
            "object": "list",
            "data": data,
            "model": AZURE_DEPLOYMENT,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                path = unquote(urlsplit(self.path).path)
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) or b"{}"
                headers = {key.lower(): value for key, value in self.headers.items()}
                authorization = headers.get("authorization", "")
                signed = _SIGV4_CREDENTIAL.match(authorization)
                if signed:
                    scheme, credential = "sigv4", signed.group("key")
                elif authorization.startswith("Bearer "):
                    scheme, credential = "bearer", authorization.removeprefix("Bearer ")
                elif "api-key" in headers:
                    scheme, credential = "api-key", headers["api-key"]
                else:
                    scheme, credential = "", ""
                for name in _AUTH_HEADERS:
                    headers.pop(name, None)
                reply = fake._serve(path, raw, headers, credential, scheme)
                if reply.drop:
                    self.close_connection = True
                    return
                data = json.dumps(reply.payload).encode()
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        return Handler

    def _serve(
        self, path: str, raw: bytes, headers: dict[str, str], credential: str, scheme: str
    ) -> _Reply:
        bedrock = _BEDROCK_PATH.match(path)
        provider = {FOUNDRY_PATH: "foundry_cohere", AZURE_PATH: "azure"}.get(
            path, "bedrock" if bedrock else ""
        )
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        model = bedrock.group("model") if bedrock else str(body.get("model", ""))
        request = ReceivedEmbeddingRequest(
            provider, path, model, headers, credential, body, scheme, self._clock()
        )
        self._record(request)
        try:
            if provider == "foundry_cohere":
                return self._foundry(request)
            if provider == "azure":
                return _Reply(200, self._azure(request))
            if provider == "bedrock":
                if scheme not in ("sigv4", "bearer") or not credential:
                    raise _RefusedError(403, {"message": "Missing SigV4 signature or bearer token"})
                return _Reply(200, self._bedrock(request))
            raise _RefusedError(404, {"error": {"code": "NotFound", "message": f"no route {path}"}})
        except _RefusedError as refused:
            return _Reply(refused.status, refused.payload, refused.headers)
