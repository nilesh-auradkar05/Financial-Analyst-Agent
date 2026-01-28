"""
FINANCIAL ANALYST AGENT SYSTEM - PROMETHEUS METRICS
"""

import time
from contextlib import contextmanager
from typing import Any, Callable

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST as PROM_CONTENT_TYPE_LATEST,
    )
    from prometheus_client import (
        Counter as PromCounter,
    )
    from prometheus_client import (
        Gauge as PromGauge,
    )
    from prometheus_client import (
        Histogram as PromHistogram,
    )
    from prometheus_client import (
        generate_latest as prom_generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    PROM_CONTENT_TYPE_LATEST = None
    PromCounter = None
    PromGauge = None
    PromHistogram = None
    prom_generate_latest = None

CONTENT_TYPE_LATEST: str
Counter: Any
Gauge: Any
Histogram: Any
generate_latest: Callable[[], bytes]

if PROMETHEUS_AVAILABLE:
    assert PROM_CONTENT_TYPE_LATEST is not None
    assert PromCounter is not None
    assert PromGauge is not None
    assert PromHistogram is not None
    assert prom_generate_latest is not None
    CONTENT_TYPE_LATEST = PROM_CONTENT_TYPE_LATEST
    Counter = PromCounter
    Gauge = PromGauge
    Histogram = PromHistogram
    generate_latest = prom_generate_latest
else:
    CONTENT_TYPE_LATEST = "text/plain"

    def _generate_latest_fallback() -> bytes:
        return b"# prometheus_client not installed\n"

    generate_latest = _generate_latest_fallback
    Counter = Gauge = Histogram = object


# =============================================================================
# ESSENTIAL METRICS
# =============================================================================

if PROMETHEUS_AVAILABLE:

    # Request metrics
    REQUEST_COUNT = Counter(
        "alpha_analyst_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )

    REQUEST_LATENCY = Histogram(
        "alpha_analyst_request_latency_seconds",
        "Request latency",
        ["endpoint"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
    )

    # Agent metrics
    AGENT_RUNS = Counter(
        "alpha_analyst_agent_runs_total",
        "Total agent runs",
        ["ticker", "status"],
    )

    AGENT_DURATION = Histogram(
        "alpha_analyst_agent_duration_seconds",
        "Agent run duration",
        buckets=[10, 30, 60, 90, 120, 180, 300],
    )

    ACTIVE_RUNS = Gauge(
        "alpha_analyst_active_runs",
        "Currently running analyses",
    )

    # Errors
    ERRORS = Counter(
        "alpha_analyst_errors_total",
        "Total errors",
        ["component"],
    )


# =============================================================================
# TRACKING CONTEXT MANAGERS
# =============================================================================


@contextmanager
def track_request(method: str, endpoint: str):
    """Track HTTP request metrics."""
    if not PROMETHEUS_AVAILABLE:
        yield
        return

    start = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)


@contextmanager
def track_agent_run(ticker: str):
    """Track agent execution metrics."""
    if not PROMETHEUS_AVAILABLE:
        yield
        return

    ACTIVE_RUNS.inc()
    start = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        AGENT_RUNS.labels(ticker=ticker.upper(), status=status).inc()
        AGENT_DURATION.observe(time.time() - start)
        ACTIVE_RUNS.dec()


def record_error(component: str):
    """Record an error."""
    if PROMETHEUS_AVAILABLE:
        ERRORS.labels(component=component).inc()


# =============================================================================
# METRICS ENDPOINT
# =============================================================================


def get_metrics() -> bytes:
    """Get metrics in Prometheus format."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n"
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint."""
    return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"
