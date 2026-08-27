"""Per-session SSE event hubs with ring-buffer replay and fan-out."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
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
THINKING_MAX_CHARS = 12_000

_AGENTS = frozenset({"extractor", "verifier", "reporter", "dialogue"})
_AGENT_STATUSES = frozenset({"started", "retrying", "completed", "degraded"})
_SEARCH_STATUSES = frozenset({"started", "completed", "failed"})

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
    thinking_enabled: bool = False
    thinking_supported: bool = False
    thinking_max_chars: int = THINKING_MAX_CHARS
    thinking_chars: int = 0
    thinking_truncated: bool = False


@dataclass(frozen=True)
class EventContext:
    """Execution context used to attach retry/search events to a run."""

    session_id: str
    agent: Literal["extractor", "verifier", "reporter", "dialogue"]
    stage: str
    claim_index: int | None = None


_event_context: ContextVar[EventContext | None] = ContextVar(
    "factcheck_event_context", default=None
)


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


def create_session_hub(
    session_id: str,
    run_id: str | None = None,
    *,
    thinking_enabled: bool = False,
    thinking_supported: bool = False,
    thinking_max_chars: int = THINKING_MAX_CHARS,
) -> SessionStreamHub:
    """Create a hub for *session_id*. Call before starting a background pipeline task."""
    existing = _hubs.get(session_id)
    if existing is not None and existing.state == "open":
        logger.warning(
            "[event_bus] Superseding open hub for session %s (run_id=%s)",
            session_id,
            existing.run_id,
        )
        _close_hub_immediately(existing)

    hub = SessionStreamHub(
        session_id=session_id,
        run_id=run_id,
        thinking_enabled=bool(thinking_enabled),
        thinking_supported=bool(thinking_supported),
        thinking_max_chars=max(1, int(thinking_max_chars)),
    )
    _hubs[session_id] = hub
    return hub


def create_session_queue(
    session_id: str,
    run_id: str | None = None,
    **kwargs: Any,
) -> SessionStreamHub:
    """Backward-compatible alias for :func:`create_session_hub`."""
    return create_session_hub(session_id, run_id=run_id, **kwargs)


async def push_event(session_id: str, event: str, data: dict[str, Any]) -> None:
    """Push an SSE event. Safe to call even if no subscriber is connected yet."""
    hub = _hubs.get(session_id)
    if hub is None or hub.state != "open":
        return

    stored = StoredEvent(event=event, data=data)
    hub.buffer.append(stored)
    for queue in list(hub.subscribers):
        await queue.put(stored)


def agent_progress_payload(
    *,
    agent: str,
    stage: str,
    status: str,
    message: str,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Build the stable, user-safe agent progress event payload."""

    if agent not in _AGENTS:
        raise ValueError("invalid agent")
    if status not in _AGENT_STATUSES:
        raise ValueError("invalid agent progress status")
    payload: dict[str, Any] = {
        "agent": agent,
        "stage": str(stage)[:80],
        "status": status,
        "message": str(message)[:240],
    }
    if attempt is not None:
        payload["attempt"] = max(1, int(attempt))
    if max_attempts is not None:
        payload["max_attempts"] = max(1, int(max_attempts))
    return payload


async def push_agent_progress(
    session_id: str,
    *,
    agent: str,
    stage: str,
    status: str,
    message: str,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    await push_event(
        session_id,
        "agent_progress",
        agent_progress_payload(
            agent=agent,
            stage=stage,
            status=status,
            message=message,
            attempt=attempt,
            max_attempts=max_attempts,
        ),
    )


def search_progress_payload(
    *,
    claim_index: int,
    query_index: int,
    total_queries: int,
    status: str,
    provider: str | None = None,
    result_count: int | None = None,
) -> dict[str, Any]:
    """Build a search progress payload without leaking query text or errors."""

    if status not in _SEARCH_STATUSES:
        raise ValueError("invalid search progress status")
    payload: dict[str, Any] = {
        "claim_index": max(0, int(claim_index)),
        "query_index": max(0, int(query_index)),
        "total_queries": max(0, int(total_queries)),
        "status": status,
    }
    if provider:
        payload["provider"] = str(provider)[:80]
    if result_count is not None:
        payload["result_count"] = max(0, int(result_count))
    return payload


async def push_search_progress(
    session_id: str,
    *,
    claim_index: int,
    query_index: int,
    total_queries: int,
    status: str,
    provider: str | None = None,
    result_count: int | None = None,
) -> None:
    await push_event(
        session_id,
        "search_progress",
        search_progress_payload(
            claim_index=claim_index,
            query_index=query_index,
            total_queries=total_queries,
            status=status,
            provider=provider,
            result_count=result_count,
        ),
    )


def thinking_chunk_payload(
    *,
    agent: str,
    stage: str,
    text: str,
    claim_index: int | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    """Build a thinking event payload; callers must still pass hub gating."""

    if agent not in _AGENTS:
        raise ValueError("invalid agent")
    payload: dict[str, Any] = {
        "agent": agent,
        "stage": str(stage)[:80],
        "text": str(text),
    }
    if claim_index is not None:
        payload["claim_index"] = max(0, int(claim_index))
    if truncated:
        payload["truncated"] = True
    return payload


async def push_thinking_chunk(
    session_id: str,
    *,
    agent: str,
    stage: str,
    text: str,
    claim_index: int | None = None,
) -> bool:
    """Publish model thinking only for a capability-confirmed opted-in run."""

    hub = _hubs.get(session_id)
    if (
        hub is None
        or hub.state != "open"
        or not hub.thinking_enabled
        or not hub.thinking_supported
        or not isinstance(text, str)
        or not text
    ):
        return False

    remaining = max(0, hub.thinking_max_chars - hub.thinking_chars)
    if remaining == 0:
        return False
    visible = text[:remaining]
    was_truncated = len(text) > len(visible)
    if visible:
        hub.thinking_chars += len(visible)
    if len(text) > len(visible):
        hub.thinking_truncated = True
    if not visible and not hub.thinking_truncated:
        return False

    await push_event(
        session_id,
        "thinking_chunk",
        thinking_chunk_payload(
            agent=agent,
            stage=stage,
            text=visible,
            claim_index=claim_index,
            truncated=was_truncated,
        ),
    )
    return True


@contextmanager
def event_scope(
    session_id: str,
    *,
    agent: Literal["extractor", "verifier", "reporter", "dialogue"],
    stage: str,
    claim_index: int | None = None,
):
    """Attach agent context to nested LLM/search calls for progress callbacks."""

    token = _event_context.set(
        EventContext(
            session_id=session_id,
            agent=agent,
            stage=stage,
            claim_index=claim_index,
        )
    )
    try:
        yield
    finally:
        _event_context.reset(token)


def current_event_context() -> EventContext | None:
    return _event_context.get()


async def on_ollama_retry(payload: dict[str, int]) -> None:
    """Retry callback used by LLM helpers; only emits safe attempt metadata."""

    context = current_event_context()
    if context is None:
        return
    attempt = payload.get("attempt", 1)
    max_attempts = payload.get("max_attempts", 1)
    await push_agent_progress(
        context.session_id,
        agent=context.agent,
        stage=context.stage,
        status="retrying",
        attempt=attempt,
        max_attempts=max_attempts,
        message=f"Model unavailable — retrying ({attempt}/{max_attempts})",
    )


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
