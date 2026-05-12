"""
FINANCIAL ANALYST AGENT SYSTEM - PROMETHEUS METRICS
"""

import time
from contextlib import contextmanager
from typing import Any, Callable


def _try_import_prometheus() -> tuple[bool, Any, Any, Any, Any, Any]:
    """Load prometheus_client or return placeholders (all Any-typed for mypy)."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST as ct  # noqa: N811
        from prometheus_client import Counter as ctr  # noqa: N813
        from prometheus_client import Gauge as gge  # noqa: N813
        from prometheus_client import Histogram as hgm  # noqa: N813
        from prometheus_client import generate_latest as gen

        return True, ct, ctr, gge, hgm, gen
    except ImportError:
        return False, None, None, None, None, None


PROMETHEUS_AVAILABLE, PROM_CONTENT_TYPE_LATEST, PromCounter, PromGauge, PromHistogram, prom_generate_latest = (
    _try_import_prometheus()
)

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
