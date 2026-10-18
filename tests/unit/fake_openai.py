"""A faithful fake of the OpenAI-compatible chat-completions wire format (Foundry, bedrock-mantle).

The real `ChatOpenAI` client builds and sends real HTTP requests; this server parses them and
computes the reply from the request, per host. No network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from langchain_openai import ChatOpenAI as _RealChatOpenAI


class FakeOpenAIServer:
    """Answers `/chat/completions` with `<model>: <last user message>`; listed hosts fail."""

    def __init__(self, failures: dict[str, int] | None = None, delay: float = 0) -> None:
        self.requests: list[httpx.Request] = []
        self.failures = failures or {}
        self.delay = delay

    def _reply(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._answer(request)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        status = self.failures.get(request.url.host)
        if status:
            return httpx.Response(
                status,
                json={"error": {"message": "unavailable", "type": "error", "code": str(status)}},
            )
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        content = f"{body['model']}: {prompt}"
        response_format = body.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            # Structured output: every property of the requested schema names the answering model.
            properties = response_format["json_schema"]["schema"]["properties"]
            content = json.dumps({name: body["model"] for name in properties})
        words = len(str(prompt).split())
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": words,
                    "completion_tokens": words + 1,
                    "total_tokens": 2 * words + 1,
                },
            },
        )

    async def _areply(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)  # recorded on arrival, so a request that times out still counts
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._answer(request)

    def chat_openai(self, **kwargs: Any) -> _RealChatOpenAI:
        """Drop-in for `langchain_openai.ChatOpenAI` that routes its HTTP traffic here."""
        return _RealChatOpenAI(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(self._reply)),
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(self._areply)),
        )

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]
