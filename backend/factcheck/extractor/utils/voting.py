"""Consensus voting helpers for extractor stages."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
ResultT = TypeVar("ResultT")

Processor = Callable[[T, object], Awaitable[tuple[bool, R | None]]]
ResultFactory = Callable[[R, T], ResultT | None]
BatchProcessor = Callable[[Sequence[T], object], Awaitable[dict[Hashable, tuple[bool, str | None]]]]


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for majority vote comparison.

    Two sentences are treated as the same vote when their normalized forms match.
    """
    return text.strip().lower().rstrip(".,;:!?").strip()


def _majority_winner(results: Sequence[str], min_successes: int) -> str | None:
    """Return the first result whose normalized value reaches quorum."""

    if not results:
        return None

    normalized_counts: Counter[str] = Counter(_normalize_for_comparison(result) for result in results)
    most_common_normalized, count = normalized_counts.most_common(1)[0]
    if count < min_successes:
        return None

    for result in results:
        if _normalize_for_comparison(result) == most_common_normalized:
            return result
    return None


def _majority_still_possible(
    results: Sequence[str],
    *,
    min_successes: int,
    remaining_attempts: int,
) -> bool:
    """Return whether future attempts can still produce a quorum."""

    if _majority_winner(results, min_successes) is not None:
        return False

    if remaining_attempts >= min_successes:
        return True

    normalized_counts: Counter[str] = Counter(_normalize_for_comparison(result) for result in results)
    best_existing = max(normalized_counts.values(), default=0)
    return best_existing + remaining_attempts >= min_successes


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


async def process_batch_with_voting(
    *,
    items: Sequence[T],
    batch_processor: BatchProcessor[T],
    llm: object,
    completions: int,
    min_successes: int,
    result_factory: ResultFactory[str, ResultT],
    item_key: Callable[[T], Hashable] | None = None,
) -> list[ResultT]:
    """Process items in batches while allowing per-item early stopping.

    Each batch attempt may return one string vote per item. An item stops
    participating once it has reached quorum or when quorum becomes
    mathematically impossible with the remaining attempts.
    """

    if not items:
        return []

    key_fn = item_key or (lambda item: item)
    keyed_items = [(key_fn(item), item) for item in items]
    item_by_key = {key: item for key, item in keyed_items}
    votes_by_key: dict[Hashable, list[str]] = {key: [] for key, _ in keyed_items}
    resolved: dict[Hashable, ResultT | None] = {}
    active_keys = [key for key, _ in keyed_items]

    for attempt in range(completions):
        if not active_keys:
            break

        active_items = [item_by_key[key] for key in active_keys]
        batch_results = await batch_processor(active_items, llm)
        remaining_attempts = completions - attempt - 1
        next_active: list[Hashable] = []

        for key in active_keys:
            vote = batch_results.get(key)
            if vote is not None:
                success, value = vote
                if success and value is not None:
                    votes_by_key[key].append(value)

            winner = _majority_winner(votes_by_key[key], min_successes)
            if winner is not None:
                resolved[key] = result_factory(winner, item_by_key[key])
                continue

            if _majority_still_possible(
                votes_by_key[key],
                min_successes=min_successes,
                remaining_attempts=remaining_attempts,
            ):
                next_active.append(key)
            else:
                logger.info(
                    "Batch voting failed: quorum impossible after %s/%s attempts",
                    attempt + 1,
                    completions,
                )

        active_keys = next_active

    return [
        resolved[key]
        for key, _item in keyed_items
        if key in resolved and resolved[key] is not None
    ]
