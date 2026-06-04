"""Consensus voting helpers for extractor stages."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
ResultT = TypeVar("ResultT")

Processor = Callable[[T, object], Awaitable[tuple[bool, R | None]]]
ResultFactory = Callable[[R, T], ResultT | None]


async def process_with_voting(
    *,
    items: Sequence[T],
    processor: Processor[T, R],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[R, ResultT],
) -> list[ResultT]:
    """Run repeated attempts per item and keep items meeting the success threshold.

    Voting attempts for the *same* item are executed **sequentially** rather
    than with ``asyncio.gather``.  Concurrent attempts would hammer a local
    Ollama instance with ``completions`` simultaneous requests for every single
    sentence, causing VRAM thrashing or OOM crashes.  Sequential execution is
    safe because the votes are independent of each other and the result is
    identical — only the wall-clock time per item increases slightly, which is
    acceptable given the hardware constraints.
    """
    results: list[ResultT] = []
    for item in items:
        # Run each voting attempt one at a time to stay within the local GPU's
        # capacity.  The semaphore in factcheck.llm.concurrency enforces the
        # global limit; this loop removes the extra concurrency pressure at the
        # call-site level.
        attempts: list[tuple[bool, R | None]] = []
        for _ in range(completions):
            attempt = await processor(item, llm)
            attempts.append(attempt)

        successes = [(success, value) for success, value in attempts if success and value is not None]
        if len(successes) < min_successes:
            logger.info("Voting rejected item with %s/%s successes", len(successes), completions)
            continue

        value = successes[0][1]
        processed = result_factory(value, item)
        if processed is not None:
            results.append(processed)

    return results
