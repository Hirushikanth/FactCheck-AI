"""Tests for the SSE event bus."""

from __future__ import annotations

import asyncio

import pytest

from factcheck.graph import event_bus


@pytest.fixture(autouse=True)
def clear_queues():
    event_bus._queues.clear()
    yield
    event_bus._queues.clear()


async def _collect_stream(session_id: str, limit: int = 10) -> list[str]:
    frames: list[str] = []
    async for frame in event_bus.stream_events(session_id):
        frames.append(frame)
        if len(frames) >= limit:
            break
    return frames


@pytest.mark.asyncio
async def test_push_and_stream_events() -> None:
    event_bus.create_session_queue("sess-1")

    async def consume() -> list[str]:
        frames: list[str] = []
        async for frame in event_bus.stream_events("sess-1"):
            frames.append(frame)
        return frames

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await event_bus.push_event("sess-1", "agent_start", {"agent": "extractor"})
    await event_bus.close_session_queue("sess-1")
    frames = await consumer

    assert len(frames) == 1
    assert "event: agent_start" in frames[0]
    assert "extractor" in frames[0]


@pytest.mark.asyncio
async def test_stream_without_queue_yields_nothing() -> None:
    frames = await _collect_stream("missing")
    assert frames == []


@pytest.mark.asyncio
async def test_close_sends_sentinel() -> None:
    event_bus.create_session_queue("sess-2")

    async def consume() -> list[str]:
        frames: list[str] = []
        async for frame in event_bus.stream_events("sess-2"):
            frames.append(frame)
        return frames

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await event_bus.push_event("sess-2", "pipeline_done", {"session_id": "sess-2"})
    await event_bus.close_session_queue("sess-2")
    frames = await consumer

    assert len(frames) == 1
    assert "pipeline_done" in frames[0]
