"""A faithful fake of the Foundry endpoints the app calls: Anthropic Messages
(`POST /anthropic/v1/messages`) and Cohere Rerank v2 (`POST /providers/cohere/v2/rerank`).

A real HTTP server on 127.0.0.1: the production client builds and sends real requests, this
server parses them and computes the reply from the request. For Messages it reads the
`<article index="N">` blocks of the user content, unescapes them and scores each text with a small
fixed lexicon. For rerank it scores each document by the fraction of query words it contains and
lists the results best first, not in input order (as the real endpoint does). It never replays an
answer supplied by a test. No network beyond loopback.
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

POSITIVE_WORDS = ("beat", "record", "surged", "growth", "rose")
NEGATIVE_WORDS = ("fell", "warned", "lawsuit", "decline", "missed")
SERVED_MODEL = "claude-sonnet-5-5"
DEPLOYMENT = "sonnet-5-5-sentiment"
MODEL_IDENTITY = "foundry:sonnet-5-5-sentiment:claude-sonnet-5-5"
MESSAGES_PATH = "/anthropic/v1/messages"
RERANK_PATH = "/providers/cohere/v2/rerank"
RERANK_DEPLOYMENT = "cohere-rerank-v5-fast"

_ARTICLE = re.compile(r'<article index="(\d+)">(.*?)</article>', re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")
_AUTH_HEADERS = ("api-key", "x-api-key", "authorization")


@dataclass
class ReceivedRequest:
    """One request as the provider saw it. Credentials are kept apart from `headers`."""

    path: str
    headers: dict[str, str]
    credential: str
    body: dict[str, Any]
    bearer: bool = False

    @property
    def system(self) -> str:
        system = self.body.get("system", "")
        if isinstance(system, list):
            return "".join(block.get("text", "") for block in system)
        return str(system)

    @property
    def user_content(self) -> str:
        (message,) = self.body["messages"]
        content = message["content"]
        if isinstance(content, list):
            return "".join(block.get("text", "") for block in content)
        return str(content)

    @property
    def articles(self) -> list[tuple[int, str]]:
        """(index, unescaped text) of every `<article>` block, in document order."""
        return [
            (int(index), html.unescape(text)) for index, text in _ARTICLE.findall(self.user_content)
        ]


def lexicon_label(text: str) -> str:
    lowered = text.lower()
    positive = sum(word in lowered for word in POSITIVE_WORDS)
    negative = sum(word in lowered for word in NEGATIVE_WORDS)
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def relevance(query: str, document: str) -> float:
    """Fraction of the query's distinct words that appear in the document, in [0, 1]."""
    wanted = set(_WORD.findall(query.lower()))
    if not wanted:
        return 0.0
    return len(wanted & set(_WORD.findall(document.lower()))) / len(wanted)


def _scores(label: str) -> dict[str, float]:
    return {name: (0.8 if name == label else 0.1) for name in ("positive", "negative", "neutral")}


class FakeFoundry:
    """Serves Messages and rerank replies computed from each request.

    `status` makes every request fail with that HTTP status; `fail_first` fails only the first N
    requests. `corrupt` edits the computed reply body before it is sent (a malformed provider);
    `raw_reply` replaces a successful reply with those bytes (a provider that is not speaking JSON).
    `delay` is seconds of latency per request.
    """

    def __init__(self) -> None:
        self.requests: list[ReceivedRequest] = []
        self.status: int | None = None
        self.fail_first = 0
        self.fail_status = 500
        self.corrupt: Callable[[dict[str, Any]], None] | None = None
        self.delay = 0.0
        self.raw_reply: bytes | None = None
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeFoundry:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _reply(self, request: ReceivedRequest) -> tuple[int, dict[str, Any] | bytes]:
        with self._lock:
            self.requests.append(request)
            attempt = len(self.requests)
        if self.delay:
            time.sleep(self.delay)
        failing = self.status or (self.fail_status if attempt <= self.fail_first else None)
        if failing:
            return failing, {
                "type": "error",
                "error": {"type": "api_error", "message": "provider unavailable"},
            }
        if request.path == RERANK_PATH:
            status, body = self._rerank(request, attempt)
        elif request.path == MESSAGES_PATH:
            status, body = 200, self._messages(request, attempt)
        else:
            return 404, {"error": {"code": "NotFound", "message": f"no route {request.path}"}}
        if status == 200:
            if self.corrupt:
                self.corrupt(body)
            if self.raw_reply is not None:
                return 200, self.raw_reply
        return status, body

    def _rerank(self, request: ReceivedRequest, attempt: int) -> tuple[int, dict[str, Any]]:
        body = request.body
        documents = body.get("documents")
        top_n = body.get("top_n")
        valid = (
            request.credential != ""
            and request.bearer
            and isinstance(body.get("model"), str)
            and body["model"] != ""
            and isinstance(body.get("query"), str)
            and isinstance(documents, list)
            and all(isinstance(text, str) for text in documents)
            and isinstance(top_n, int)
            and 1 <= top_n <= len(documents)
        )
        if not valid:
            return 400, {"error": {"code": "BadRequest", "message": "invalid rerank request"}}
        assert documents is not None
        scored = [
            {"index": index, "relevance_score": relevance(body["query"], text)}
            for index, text in enumerate(documents)
        ]
        scored.sort(key=lambda row: (-row["relevance_score"], row["index"]))
        return 200, {
            "id": f"rerank_{attempt}",
            "results": scored[: body["top_n"]],
            "meta": {
                "api_version": {"version": "2"},
                "billed_units": {"input_tokens": sum(len(text.split()) for text in documents)},
            },
        }

    def _messages(self, request: ReceivedRequest, attempt: int) -> dict[str, Any]:
        results = [
            {"index": index, "label": label, "scores": _scores(label)}
            for index, text in request.articles
            for label in [lexicon_label(text)]
        ]
        return {
            "id": f"msg_{attempt}",
            "type": "message",
            "role": "assistant",
            "model": SERVED_MODEL,
            "content": [{"type": "text", "text": json.dumps({"results": results})}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(request.user_content.split()),
                "output_tokens": len(results),
            },
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {key.lower(): value for key, value in self.headers.items()}
                credential = next(
                    (
                        headers[name].removeprefix("Bearer ")
                        for name in _AUTH_HEADERS
                        if name in headers
                    ),
                    "",
                )
                bearer = headers.get("authorization", "").startswith("Bearer ")
                for name in _AUTH_HEADERS:
                    headers.pop(name, None)
                status, payload = fake._reply(
                    ReceivedRequest(self.path, headers, credential, body, bearer)
                )
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        return Handler
