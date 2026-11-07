"""A faithful fake of the embedding endpoints the four embedding arms call (test-plan §21).

One real HTTP server on 127.0.0.1 serves three wire formats, so the production clients build and
send real requests and this server parses them:

- Ollama `POST /api/embed` (model `qwen3-embedding:4b`, native 2560 dimensions).
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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

OLLAMA_MODEL = "qwen3-embedding:4b"
OLLAMA_NATIVE_DIMENSION = 2560
TITAN_MODEL = "amazon.titan-embed-text-v2:0"
COHERE_MODEL = "cohere.embed-v4:0"
AZURE_DEPLOYMENT = "embed-3-small"
AZURE_NATIVE_DIMENSION = 1536

OLLAMA_PATH = "/api/embed"
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


class _RefusedError(Exception):
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        super().__init__(status)
        self.status = status
        self.payload = payload


def _bad_request(message: str) -> _RefusedError:
    return _RefusedError(400, {"error": {"code": "BadRequest", "message": message}})


def _texts(value: Any) -> list[str]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(t, str) for t in items):
        raise _bad_request("input must be a non-empty string or list of strings")
    return items


class FakeEmbeddingProviders:
    """Serves Ollama, Bedrock (Titan V2, Cohere v4) and Azure embedding replies computed from each request."""

    def __init__(self) -> None:
        self.requests: list[ReceivedEmbeddingRequest] = []
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

    def requests_to(self, provider: str) -> list[ReceivedEmbeddingRequest]:
        return [request for request in self.requests if request.provider == provider]

    def _record(self, request: ReceivedEmbeddingRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def _ollama(self, request: ReceivedEmbeddingRequest) -> dict[str, Any]:
        body = request.body
        if body.get("model") != OLLAMA_MODEL:
            raise _RefusedError(404, {"error": f"model {body.get('model')!r} not found"})
        dimension = body.get("dimensions") or OLLAMA_NATIVE_DIMENSION
        texts = _texts(body.get("input"))
        return {
            "model": OLLAMA_MODEL,
            "embeddings": [vector_for(OLLAMA_MODEL, text, dimension) for text in texts],
            "total_duration": 1,
            "prompt_eval_count": sum(len(text.split()) for text in texts),
        }

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
                status, payload = fake._serve(path, raw, headers, credential, scheme)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        return Handler

    def _serve(
        self, path: str, raw: bytes, headers: dict[str, str], credential: str, scheme: str
    ) -> tuple[int, dict[str, Any]]:
        bedrock = _BEDROCK_PATH.match(path)
        provider = {OLLAMA_PATH: "ollama", AZURE_PATH: "azure"}.get(
            path, "bedrock" if bedrock else ""
        )
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        model = bedrock.group("model") if bedrock else str(body.get("model", ""))
        request = ReceivedEmbeddingRequest(provider, path, model, headers, credential, body, scheme)
        self._record(request)
        try:
            if provider == "ollama":
                return 200, self._ollama(request)
            if provider == "azure":
                return 200, self._azure(request)
            if provider == "bedrock":
                if scheme not in ("sigv4", "bearer") or not credential:
                    raise _RefusedError(403, {"message": "Missing SigV4 signature or bearer token"})
                return 200, self._bedrock(request)
            raise _RefusedError(404, {"error": {"code": "NotFound", "message": f"no route {path}"}})
        except _RefusedError as refused:
            return refused.status, refused.payload
