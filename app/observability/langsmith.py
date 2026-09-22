"""Native LangSmith boundaries and HTTP correlation; export is always optional."""
from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, ParamSpec, TypeVar, cast
from uuid import UUID, uuid4

from langsmith import Client, get_current_run_tree, trace, tracing_context
from langsmith import traceable as _traceable
from langsmith.run_trees import RunTree
from loguru import logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings
from app.llm.provider import model_metadata

_P = ParamSpec("_P")
_R = TypeVar("_R")

_client: Client | None = None
FLUSH_TIMEOUT_SECONDS = 2.0


def app_traceable(*args: Any, **kwargs: Any) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Trace app spans with a safe failure marker instead of exception details."""
    kwargs["exceptions_to_handle"] = (Exception,)
    traced = _traceable(*args, **kwargs)

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def async_wrapper(*function_args: Any, **function_kwargs: Any) -> Any:
                try:
                    return await function(*function_args, **function_kwargs)
                except Exception:
                    if run := get_current_run_tree():
                        run.metadata.update(outcome="failed", error_code="app_stage_failed")
                    raise

            return cast(Callable[_P, _R], traced(async_wrapper))

        @functools.wraps(function)
        def wrapper(*function_args: Any, **function_kwargs: Any) -> Any:
            try:
                return function(*function_args, **function_kwargs)
            except Exception:
                if run := get_current_run_tree():
                    run.metadata.update(outcome="failed", error_code="app_stage_failed")
                raise

        return cast(Callable[_P, _R], traced(wrapper))

    return decorate


def is_tracing_enabled() -> bool:
    return bool(settings.langsmith.tracing_v2 and (settings.langsmith.api_key or "").strip())


def setup_langsmith_env() -> bool:
    """Initialize one client per application lifespan, without mutating process secrets."""
    global _client
    _client = None
    if is_tracing_enabled():
        try:
            _client = Client(api_key=settings.langsmith.api_key)
        except Exception:
            logger.warning("Trace exporter initialization failed; tracing disabled")
    return _client is not None


async def shutdown_langsmith() -> None:
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(client.flush, timeout=FLUSH_TIMEOUT_SECONDS),
                timeout=FLUSH_TIMEOUT_SECONDS + 0.1,
            )
        except Exception:
            logger.warning("Trace exporter flush incomplete")


def get_client() -> Client | None:
    return _client


async def check_langsmith_connection() -> dict[str, Any]:
    # Health does not perform an authenticated network request on every HTTP poll.
    return {"available": True, "enabled": is_tracing_enabled(),
            "connected": False, "configured": _client is not None,
            "check": "configuration_only"}


@asynccontextmanager
async def trace_boundary(
    name: str, *, metadata: dict[str, Any], parent: Mapping[str, str] | None = None,
) -> AsyncIterator[RunTree | None]:
    """Keep exporter failures and exception text out of application boundaries."""
    client = get_client()
    enabled = is_tracing_enabled() and client is not None
    carrier: dict[str | bytes, str | bytes] = {key: value for key, value in (parent or {}).items()}
    restored_parent = RunTree.from_headers(carrier, client=client) if carrier and enabled else None
    with tracing_context(enabled=enabled, client=client, project_name=settings.langsmith.project,
                         parent=restored_parent or False):
        manager = trace(name, inputs={}, metadata={
            "agent": "financial_analyst", "agent_version": "1.1.0",
            "available_tools": ["search_company_news", "get_stock_data", "retrieve_filings", "analyze_sentiment_batch", "verify_memo"],
            **model_metadata(settings), **metadata,
        }, client=client)
        run = None
        if enabled:
            try:
                run = await manager.__aenter__()
            except Exception:
                logger.warning("Trace exporter unavailable")
        # Explicit false also blocks native callbacks when startup/export setup failed.
        with tracing_context(enabled=enabled and run is not None, client=client,
                             project_name=settings.langsmith.project, parent=run or False):
            try:
                yield run
            finally:
                if run is not None:
                    try:
                        await manager.__aexit__(None, None, None)
                    except Exception:
                        logger.warning("Trace exporter finalization failed")


class RequestTracingMiddleware:
    """End HTTP timing at the response body, before Starlette background jobs."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).getlist("x-request-id")
        request_id = str(uuid4())
        invalid = False
        if supplied:
            try:
                if len(supplied) != 1 or len(supplied[0]) != 36 or str(UUID(supplied[0])) != supplied[0]:
                    raise ValueError
                request_id = supplied[0]
            except ValueError:
                invalid = True
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        started = time.monotonic()
        status_code = 500
        sent = False
        async with trace_boundary("HTTP " + scope["method"], metadata={
            "request_id": request_id, "method": scope["method"], "app_version": "1.1.0",
        }) as run:
            state["trace_id"] = str(run.trace_id) if run else None
            state["trace_parent"] = run.to_headers() if run else None

            async def traced_send(message: Message) -> None:
                nonlocal status_code, sent
                if message["type"] == "http.response.start":
                    sent = True
                    status_code = message["status"]
                    headers = MutableHeaders(scope=message)
                    headers["X-Request-ID"] = request_id
                    if run:
                        headers["X-Trace-ID"] = str(run.trace_id)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    route = getattr(scope.get("route"), "path", "unmatched")
                    outcome = "error" if status_code >= 400 else state.get("trace_outcome", "success")
                    if run:
                        run.name = f"HTTP {scope['method']} {route}"
                        run.metadata.update({"route": route, "status_code": status_code,
                                             "duration_ms": (time.monotonic() - started) * 1000,
                                             "outcome": outcome, **state.get("trace_error", {})})
                        run.end(outputs={"status_code": status_code, "outcome": outcome},
                                error=state.get("trace_error", {}).get("code") if status_code >= 500 else None)
                    logger.info("HTTP request | request_id={} | trace_id={} | route={} | status={}",
                                request_id, state["trace_id"], route, status_code)
                await send(message)

            try:
                if invalid:
                    error = {"code": "invalid_request_id", "message": "X-Request-ID must be a canonical UUID.", "error_id": str(uuid4())}
                    state["trace_error"] = {key: error[key] for key in ("code", "error_id")}
                    await JSONResponse({"error": error}, status_code=400, headers={"Cache-Control": "no-store"})(scope, receive, traced_send)
                else:
                    await self.app(scope, receive, traced_send)
            except Exception:
                if sent:
                    raise
                error = {"code": "internal_error", "message": "Internal server error.", "error_id": str(uuid4())}
                state["trace_error"] = {key: error[key] for key in ("code", "error_id")}
                await JSONResponse({"error": error}, status_code=500, headers={"Cache-Control": "no-store"})(scope, receive, traced_send)
