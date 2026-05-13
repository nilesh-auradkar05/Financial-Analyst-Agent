from __future__ import annotations

import pytest

from evaluation.run_rag_quality_eval import (
    _run_with_retries,
    _run_with_retries_async,
)


def test_run_with_retries_retries_retryable_errors() -> None:
    attempts = 0

    def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    result = _run_with_retries(
        flaky_call,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    assert result == "ok"
    assert attempts == 3


def test_run_with_retries_does_not_retry_non_retryable_errors() -> None:
    attempts = 0

    def failing_call() -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("schema validation failed")

    with pytest.raises(RuntimeError, match="schema validation failed"):
        _run_with_retries(
            failing_call,
            max_retries=2,
            retry_backoff_seconds=0,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_run_with_retries_async_retries_retryable_errors() -> None:
    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("timeout")
        return "ok"

    result = await _run_with_retries_async(
        lambda: flaky_call(),
        max_retries=1,
        retry_backoff_seconds=0,
    )

    assert result == "ok"
    assert attempts == 2


