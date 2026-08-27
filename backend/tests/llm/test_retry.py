from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from factcheck.llm.retry import invoke_with_ollama_retry


async def test_retries_transient_network_error_and_reports_next_attempt() -> None:
    attempts = 0
    retries: list[dict[str, int]] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("Ollama unavailable")
        return "ok"

    result = await invoke_with_ollama_retry(
        operation,
        max_attempts=3,
        on_retry=retries.append,
        sleep=lambda _: asyncio.sleep(0),
        random_fn=lambda: 0.0,
    )

    assert result == "ok"
    assert attempts == 2
    assert retries == [{"attempt": 2, "max_attempts": 3}]


async def test_exhaustion_reraises_after_three_transient_failures() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("Ollama timed out")

    with pytest.raises(httpx.ReadTimeout):
        await invoke_with_ollama_retry(
            operation,
            max_attempts=3,
            sleep=lambda _: asyncio.sleep(0),
            random_fn=lambda: 0.0,
        )

    assert attempts == 3


async def test_backoff_is_exponential_and_capped_including_jitter() -> None:
    delays: list[float] = []

    async def operation() -> None:
        raise httpx.ConnectError("Ollama unavailable")

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(httpx.ConnectError):
        await invoke_with_ollama_retry(
            operation,
            max_attempts=7,
            sleep=record_sleep,
            random_fn=lambda: 1.0,
        )

    assert delays == [0.3125, 0.625, 1.25, 2.5, 5.0, 8.0]


@pytest.mark.parametrize(
    "error",
    [
        ValidationError.from_exception_data("Demo", [{"type": "missing", "loc": ("value",)}]),
        asyncio.CancelledError(),
        ValueError("malformed output"),
    ],
)
async def test_non_transport_errors_are_not_retried(error: BaseException) -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error)):
        await invoke_with_ollama_retry(
            operation,
            max_attempts=3,
            sleep=lambda _: asyncio.sleep(0),
            random_fn=lambda: 0.0,
        )

    assert attempts == 1
