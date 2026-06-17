"""Per-session SSE event hubs with ring-buffer replay and fan-out."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from factcheck.streaming.sse import format_sse

logger = logging.getLogger(__name__)

BUFFER_MAXLEN = 256
REPLAY_TTL_SECONDS = 120
WAIT_FOR_HUB_TIMEOUT = 5.0
PING_INTERVAL_SECONDS = 30.0
_WAIT_POLL_INTERVAL = 0.05

_SENTINEL = None

_hubs: dict[str, SessionStreamHub] = {}


@dataclass(frozen=True)
class StoredEvent:
    event: str
    data: dict[str, Any]


@dataclass
class SessionStreamHub:
    session_id: str
    run_id: str | None
    buffer: deque[StoredEvent] = field(default_factory=lambda: deque(maxlen=BUFFER_MAXLEN))
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    state: Literal["open", "closed"] = "open"
    closed_at: float | None = None


class StreamUnavailable(Exception):
    """Raised when no SSE hub is available for subscription."""

    def __init__(
        self,
        code: str,
        *,
        session_status: str,
        active_run_id: str | None = None,
    ) -> None:
        self.code = code
        self.session_status = session_status
        self.active_run_id = active_run_id
        super().__init__(code)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evict_if_expired(hub: SessionStreamHub) -> SessionStreamHub | None:
    if hub.state == "closed" and hub.closed_at is not None:
        if time.monotonic() - hub.closed_at > REPLAY_TTL_SECONDS:
            _hubs.pop(hub.session_id, None)
            return None
    return hub


def get_hub(session_id: str) -> SessionStreamHub | None:
    """Return an open hub or a closed hub still within the replay TTL."""
    hub = _hubs.get(session_id)
    if hub is None:
        return None
    return _evict_if_expired(hub)


def _close_hub_immediately(hub: SessionStreamHub) -> None:
    hub.state = "closed"
    hub.closed_at = time.monotonic()
    for queue in list(hub.subscribers):
        queue.put_nowait(_SENTINEL)


def create_session_hub(session_id: str, run_id: str | None = None) -> SessionStreamHub:
    """Create a hub for *session_id*. Call before starting a background pipeline task."""
    existing = _hubs.get(session_id)
    if existing is not None and existing.state == "open":
        logger.warning(
            "[event_bus] Superseding open hub for session %s (run_id=%s)",
            session_id,
            existing.run_id,
        )
        _close_hub_immediately(existing)

    hub = SessionStreamHub(session_id=session_id, run_id=run_id)
    _hubs[session_id] = hub
    return hub


def create_session_queue(session_id: str, run_id: str | None = None) -> SessionStreamHub:
    """Backward-compatible alias for :func:`create_session_hub`."""
    return create_session_hub(session_id, run_id=run_id)


async def push_event(session_id: str, event: str, data: dict[str, Any]) -> None:
    """Push an SSE event. Safe to call even if no subscriber is connected yet."""
    hub = _hubs.get(session_id)
    if hub is None or hub.state != "open":
        return

    stored = StoredEvent(event=event, data=data)
    hub.buffer.append(stored)
    for queue in list(hub.subscribers):
        await queue.put(stored)


async def close_session_hub(session_id: str) -> None:
    """Close the hub, notify subscribers, and retain buffer for TTL replay."""
    hub = _hubs.get(session_id)
    if hub is None:
        return

    _close_hub_immediately(hub)


async def close_session_queue(session_id: str) -> None:
    """Backward-compatible alias for :func:`close_session_hub`."""
    await close_session_hub(session_id)


async def wait_for_hub(
    session_id: str,
    timeout: float = WAIT_FOR_HUB_TIMEOUT,
) -> SessionStreamHub | None:
    """Poll until a hub appears or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hub = get_hub(session_id)
        if hub is not None:
            return hub
        await asyncio.sleep(_WAIT_POLL_INTERVAL)
    return None


async def resolve_stream(session_id: str, session: dict[str, Any]) -> SessionStreamHub:
    """Return a subscribable hub or raise :class:`StreamUnavailable`."""
    hub = get_hub(session_id)
    if hub is not None:
        return hub

    status = session.get("status", "")
    active_run_id = session.get("active_run_id")

    if status == "running":
        hub = await wait_for_hub(session_id)
        if hub is not None:
            return hub
        raise StreamUnavailable(
            "pipeline_orphaned",
            session_status=status,
            active_run_id=active_run_id,
        )

    raise StreamUnavailable(
        "stream_missed",
        session_status=status,
        active_run_id=active_run_id,
    )


async def stream_events(session_id: str) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for one session."""
    hub = get_hub(session_id)
    if hub is None:
        return

    replay_snapshot = list(hub.buffer)
    subscriber: asyncio.Queue = asyncio.Queue()
    hub.subscribers.add(subscriber)

    try:
        yield format_sse(
            "stream_open",
            {
                "session_id": hub.session_id,
                "run_id": hub.run_id,
                "replay_count": len(replay_snapshot),
                "hub_state": hub.state,
                "server_time": _now_iso(),
            },
        )

        for stored in replay_snapshot:
            yield format_sse(stored.event, stored.data)

        if hub.state == "closed":
            return

        while True:
            try:
                item = await asyncio.wait_for(
                    subscriber.get(),
                    timeout=PING_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue

            if item is _SENTINEL:
                break

            if isinstance(item, StoredEvent):
                yield format_sse(item.event, item.data)
    finally:
        hub.subscribers.discard(subscriber)
