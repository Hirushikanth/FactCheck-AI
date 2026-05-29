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


async def process_with_voting(
    *,
    items: Sequence[T],
    processor: Processor[T, R],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[R, ResultT],
) -> list[ResultT]:
    """Run repeated attempts per item and keep items meeting the success threshold."""

    results: list[ResultT] = []
    for item in items:
        attempts = await asyncio.gather(*(processor(item, llm) for _ in range(completions)))
        successes = [(success, value) for success, value in attempts if success and value is not None]
        if len(successes) < min_successes:
            logger.info("Voting rejected item with %s/%s successes", len(successes), completions)
            continue

        value = successes[0][1]
        processed = result_factory(value, item)
        if processed is not None:
            results.append(processed)

    return results
