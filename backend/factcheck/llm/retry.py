"""Bounded retry support for transient local Ollama failures."""

from __future__ import annotations

import asyncio
import inspect
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx


T = TypeVar("T")
RetryCallback = Callable[[dict[str, int]], object | Awaitable[object]]
Sleep = Callable[[float], Awaitable[object]]
RandomFn = Callable[[], float]

# Ollama can briefly reject requests while loading a model or under pressure.
# Validation and malformed-output errors are intentionally excluded.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_retryable(error: Exception) -> bool:
    """Return whether an exception represents a transient transport failure."""

    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code in _RETRYABLE_STATUS_CODES
    )


async def invoke_with_ollama_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    on_retry: RetryCallback | None = None,
    sleep: Sleep = asyncio.sleep,
    random_fn: RandomFn = random.random,
) -> T:
    """Invoke an async Ollama operation with bounded transient-error retries.

    ``max_attempts`` is the total number of calls, including the initial one.
    The callback is called immediately before each retry with a safe payload.
    Backoff is capped and includes a small jitter to avoid synchronized retries.
    """

    attempts_limit = max(1, int(max_attempts))
    for attempt in range(1, attempts_limit + 1):
        try:
            return await operation()
        except Exception as error:
            if not _is_retryable(error) or attempt >= attempts_limit:
                raise

            next_attempt = attempt + 1
            if on_retry is not None:
                callback_result = on_retry(
                    {"attempt": next_attempt, "max_attempts": attempts_limit}
                )
                if inspect.isawaitable(callback_result):
                    await callback_result

            # 0.25, 0.5, 1, 2... seconds, capped at 8 seconds, with 0-25%
            # jitter. Tests inject sleep/random_fn so this remains deterministic.
            delay = min(8.0, 0.25 * (2 ** (attempt - 1)))
            delay = min(8.0, delay + delay * 0.25 * max(0.0, min(1.0, random_fn())))
            await sleep(delay)

    raise RuntimeError("unreachable retry state")
