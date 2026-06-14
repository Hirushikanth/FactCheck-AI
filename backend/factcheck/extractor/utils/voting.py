"""Consensus voting helpers for extractor stages."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
ResultT = TypeVar("ResultT")

Processor = Callable[[T, object], Awaitable[tuple[bool, R | None]]]
ResultFactory = Callable[[R, T], ResultT | None]


async def _vote_single_item(
    *,
    item: T,
    processor: Processor[T, R],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[R, ResultT],
) -> ResultT | None:
    """Run repeated attempts for one item and return a result if the threshold is met.

    Attempts stop early once ``min_successes`` is reached.  GPU safety is
    enforced by :func:`factcheck.llm.concurrency.get_ollama_semaphore` inside
    each processor call, not by serialising attempts here.
    """
    successes: list[R] = []
    for _ in range(completions):
        success, value = await processor(item, llm)
        if success and value is not None:
            successes.append(value)
            if len(successes) >= min_successes:
                break

    if len(successes) < min_successes:
        logger.info("Voting rejected item with %s/%s successes", len(successes), completions)
        return None

    processed = result_factory(successes[0], item)
    return processed


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

    Items are scheduled concurrently via ``asyncio.gather``.  The global
    Ollama semaphore in :func:`factcheck.llm.concurrency.get_ollama_semaphore`
    caps simultaneous in-flight requests, so this does not overload local
    hardware.  Within each item, attempts stop early once ``min_successes``
    is reached (e.g. ``completions=3`` with ``min_successes=2`` may issue
    only two calls on the happy path).
    """
    gathered = await asyncio.gather(
        *(
            _vote_single_item(
                item=item,
                processor=processor,
                llm=llm,
                completions=completions,
                min_successes=min_successes,
                result_factory=result_factory,
            )
            for item in items
        )
    )
    return [result for result in gathered if result is not None]
