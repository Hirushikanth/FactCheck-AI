"""Per-session asyncio queues for SSE pipeline event streaming."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from factcheck.streaming.sse import format_sse

_queues: dict[str, asyncio.Queue] = {}

_SENTINEL = None


def create_session_queue(session_id: str) -> asyncio.Queue:
    """Create a queue for *session_id*. Call before starting a background pipeline task."""
    queue: asyncio.Queue = asyncio.Queue()
    _queues[session_id] = queue
    return queue


async def push_event(session_id: str, event: str, data: dict[str, Any]) -> None:
    """Push an SSE event. Safe to call even if no subscriber is connected yet."""
    queue = _queues.get(session_id)
    if queue is not None:
        await queue.put({"event": event, "data": data})


async def close_session_queue(session_id: str) -> None:
    """Send sentinel and remove the queue after pipeline completes or errors."""
    queue = _queues.pop(session_id, None)
    if queue is not None:
        await queue.put(_SENTINEL)


async def stream_events(session_id: str) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for one session."""
    queue = _queues.get(session_id)
    if queue is None:
        return

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue

        if item is _SENTINEL:
            break

        yield format_sse(item["event"], item["data"])
