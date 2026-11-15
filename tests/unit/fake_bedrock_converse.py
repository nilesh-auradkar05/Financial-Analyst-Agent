"""A faithful fake of the Bedrock runtime Converse API (`POST /model/{modelId}/converse`).

A real HTTP server on 127.0.0.1: a production boto3 client pointed at it with
`AWS_ENDPOINT_URL_BEDROCK_RUNTIME` builds and signs real requests; this server parses them and
computes the reply from the request. A request that carries `toolConfig` gets a `toolUse` block for
its first tool, the input filled from that tool's input schema (every string field holds the model
id of the request, every other field its type's zero); a plain request gets the text
`"<model id>: <prompt>"`. It never replays an answer supplied by a test. `status` makes every
request fail with that HTTP status. No network beyond loopback.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

_CONVERSE_PATH = re.compile(r"^/model/(?P<model>[^/]+)/converse$")
_ERROR_TYPES = {
    401: "UnrecognizedClientException",
    429: "ThrottlingException",
    500: "InternalServerException",
}


@dataclass
class ReceivedConverse:
    """One Converse request as the runtime saw it."""

    path: str
    model_id: str
    headers: dict[str, str]
    body: dict[str, Any]

    @property
    def bearer(self) -> str:
        """The bearer token of the `Authorization` header, empty when it is not a bearer one."""
        authorization = self.headers.get("authorization", "")
        return authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""

    @property
    def prompt(self) -> str:
        """The text of the last user message."""
        message = self.body["messages"][-1]
        return "".join(block.get("text", "") for block in message["content"])


def _zero_for(json_type: object, text: str) -> object:
    return {
        "string": text,
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(str(json_type), text)


class FakeBedrockConverse:
    def __init__(self) -> None:
        self.requests: list[ReceivedConverse] = []
        self.status: int | None = None
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _serve(self) -> None:
        self._server.serve_forever(poll_interval=0.02)

    def start(self) -> FakeBedrockConverse:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _reply(self, request: ReceivedConverse) -> tuple[int, dict[str, Any], dict[str, str]]:
        with self._lock:
            self.requests.append(request)
            attempt = len(self.requests)
        if self.status:
            error = _ERROR_TYPES.get(self.status, "InternalServerException")
            return self.status, {"message": "runtime unavailable"}, {"x-amzn-ErrorType": error}
        tools = request.body.get("toolConfig", {}).get("tools", [])
        if tools:
            spec = tools[0]["toolSpec"]
            properties = spec.get("inputSchema", {}).get("json", {}).get("properties", {})
            content: list[dict[str, Any]] = [
                {
                    "toolUse": {
                        "toolUseId": f"tooluse_{attempt}",
                        "name": spec["name"],
                        "input": {
                            name: _zero_for(field.get("type"), request.model_id)
                            for name, field in properties.items()
                        },
                    }
                }
            ]
            stop_reason = "tool_use"
        else:
            content = [{"text": f"{request.model_id}: {request.prompt}"}]
            stop_reason = "end_turn"
        return (
            200,
            {
                "output": {"message": {"role": "assistant", "content": content}},
                "stopReason": stop_reason,
                "usage": {
                    "inputTokens": len(request.prompt.split()),
                    "outputTokens": 1,
                    "totalTokens": len(request.prompt.split()) + 1,
                },
                "metrics": {"latencyMs": 1},
            },
            {},
        )

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {key.lower(): value for key, value in self.headers.items()}
                path = unquote(urlsplit(self.path).path)
                matched = _CONVERSE_PATH.match(path)
                extra: dict[str, str]
                if not matched:
                    status, payload, extra = 404, {"message": f"no route {path}"}, {}
                else:
                    status, payload, extra = fake._reply(
                        ReceivedConverse(path, matched.group("model"), headers, body)
                    )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        return Handler
