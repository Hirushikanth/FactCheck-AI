"""Consensus voting helpers for extractor stages."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
ResultT = TypeVar("ResultT")

Processor = Callable[[T, object], Awaitable[tuple[bool, R | None]]]
ResultFactory = Callable[[R, T], ResultT | None]


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for majority vote comparison.

    Two sentences are treated as the same vote when their normalized forms match.
    """
    return text.strip().lower().rstrip(".,;:!?").strip()


async def _majority_vote_single_item(
    *,
    item: T,
    processor: Processor[T, R],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[R, ResultT],
) -> ResultT | None:
    """Run all completions, then pick the majority result.

    Unlike threshold-based early stopping, this runs every completion and
    accepts the output only when at least ``min_successes`` responses agree.
    """
    all_results: list[R] = []

    for _ in range(completions):
        success, value = await processor(item, llm)
        if success and value is not None:
            all_results.append(value)

    if len(all_results) < min_successes:
        logger.info(
            "Voting failed: only %s/%s successful responses",
            len(all_results),
            completions,
        )
        return None

    if all_results and isinstance(all_results[0], str):
        normalized_counts: Counter[str] = Counter(
            _normalize_for_comparison(str(r)) for r in all_results
        )
        most_common_normalized, count = normalized_counts.most_common(1)[0]
        if count < min_successes:
            logger.info(
                "No majority: best was %s/%s for '%s'",
                count,
                completions,
                most_common_normalized[:50],
            )
            return None
        for r in all_results:
            if _normalize_for_comparison(str(r)) == most_common_normalized:
                return result_factory(r, item)

    if len(all_results) >= min_successes:
        return result_factory(all_results[0], item)

    return None


async def process_with_voting(
    *,
    items: Sequence[T],
    processor: Processor[T, R],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[R, ResultT],
) -> list[ResultT]:
    """Process items with majority voting across multiple completions.

    All completions run before voting occurs. The most common output wins if it
    meets the ``min_successes`` threshold. For string outputs, normalized string
    comparison determines majority. For structured outputs, falls back to
    threshold-based selection.
    """
    gathered = await asyncio.gather(
        *(
            _majority_vote_single_item(
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
