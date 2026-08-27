"""Tests for the SSE event hub."""

from __future__ import annotations

import asyncio

import pytest

from factcheck.graph import event_bus


@pytest.fixture(autouse=True)
def clear_hubs():
    event_bus._hubs.clear()
    yield
    event_bus._hubs.clear()


async def _collect_stream(session_id: str, limit: int = 10) -> list[str]:
    frames: list[str] = []
    async for frame in event_bus.stream_events(session_id):
        frames.append(frame)
        if len(frames) >= limit:
            break
    return frames


@pytest.mark.asyncio
async def test_push_and_stream_events() -> None:
    event_bus.create_session_hub("sess-1", run_id="run-1")

    async def consume() -> list[str]:
        frames: list[str] = []
        async for frame in event_bus.stream_events("sess-1"):
            frames.append(frame)
        return frames

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await event_bus.push_event("sess-1", "agent_start", {"agent": "extractor"})
    await event_bus.close_session_hub("sess-1")
    frames = await consumer

    assert len(frames) == 2
    assert "event: stream_open" in frames[0]
    assert "event: agent_start" in frames[1]
    assert "extractor" in frames[1]


@pytest.mark.asyncio
async def test_resolve_stream_raises_when_no_hub() -> None:
    with pytest.raises(event_bus.StreamUnavailable) as exc_info:
        await event_bus.resolve_stream(
            "missing",
            {"status": "done", "active_run_id": None},
        )
    assert exc_info.value.code == "stream_missed"


@pytest.mark.asyncio
async def test_resolve_stream_raises_pipeline_orphaned(monkeypatch) -> None:
    monkeypatch.setattr(event_bus, "WAIT_FOR_HUB_TIMEOUT", 0.05)
    with pytest.raises(event_bus.StreamUnavailable) as exc_info:
        await event_bus.resolve_stream(
            "sess-running",
            {"status": "running", "active_run_id": "run-1"},
        )
    assert exc_info.value.code == "pipeline_orphaned"


@pytest.mark.asyncio
async def test_close_sends_sentinel() -> None:
    event_bus.create_session_hub("sess-2")

    async def consume() -> list[str]:
        frames: list[str] = []
        async for frame in event_bus.stream_events("sess-2"):
            frames.append(frame)
        return frames

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await event_bus.push_event("sess-2", "pipeline_done", {"session_id": "sess-2"})
    await event_bus.close_session_hub("sess-2")
    frames = await consumer

    assert len(frames) == 2
    assert "stream_open" in frames[0]
    assert "pipeline_done" in frames[1]


@pytest.mark.asyncio
async def test_late_subscriber_replays_buffered_events() -> None:
    event_bus.create_session_hub("sess-late")
    await event_bus.push_event("sess-late", "claim_found", {"claim": "Earth is round."})
    await event_bus.push_event("sess-late", "verdict_ready", {"verdict": "SUPPORTED"})
    await event_bus.close_session_hub("sess-late")

    frames: list[str] = []
    async for frame in event_bus.stream_events("sess-late"):
        frames.append(frame)

    assert len(frames) == 3
    assert "stream_open" in frames[0]
    assert "claim_found" in frames[1]
    assert "verdict_ready" in frames[2]


@pytest.mark.asyncio
async def test_subscriber_after_close_replays_within_ttl(monkeypatch) -> None:
    event_bus.create_session_hub("sess-replay")
    await event_bus.push_event("sess-replay", "pipeline_done", {"session_id": "sess-replay"})
    await event_bus.close_session_hub("sess-replay")

    frames = await _collect_stream("sess-replay", limit=5)

    assert len(frames) == 2
    assert "stream_open" in frames[0]
    assert "pipeline_done" in frames[1]


@pytest.mark.asyncio
async def test_expired_closed_hub_evicted(monkeypatch) -> None:
    monkeypatch.setattr(event_bus, "REPLAY_TTL_SECONDS", 0.01)
    event_bus.create_session_hub("sess-expired")
    await event_bus.close_session_hub("sess-expired")
    await asyncio.sleep(0.02)

    assert event_bus.get_hub("sess-expired") is None


@pytest.mark.asyncio
async def test_wait_for_hub_succeeds_when_hub_created_late() -> None:
    async def create_later() -> None:
        await asyncio.sleep(0.1)
        event_bus.create_session_hub("sess-wait")

    creator = asyncio.create_task(create_later())
    hub = await event_bus.wait_for_hub("sess-wait", timeout=1.0)
    await creator

    assert hub is not None
    assert hub.session_id == "sess-wait"


@pytest.mark.asyncio
async def test_resolve_stream_waits_for_hub() -> None:
    async def create_later() -> None:
        await asyncio.sleep(0.05)
        event_bus.create_session_hub("sess-resolve")

    creator = asyncio.create_task(create_later())
    hub = await event_bus.resolve_stream(
        "sess-resolve",
        {"status": "running", "active_run_id": "run-1"},
    )
    await creator

    assert hub.session_id == "sess-resolve"


@pytest.mark.asyncio
async def test_progress_events_replay_in_order_with_contract_fields() -> None:
    event_bus.create_session_hub("sess-contract")

    await event_bus.push_agent_progress(
        "sess-contract",
        agent="verifier",
        stage="claim_verification",
        status="started",
        message="Verifier is checking claims",
    )
    await event_bus.push_search_progress(
        "sess-contract",
        claim_index=0,
        query_index=0,
        total_queries=2,
        provider="duckduckgo",
        status="completed",
        result_count=3,
    )
    await event_bus.close_session_hub("sess-contract")

    frames = await _collect_stream("sess-contract", limit=5)
    assert "event: agent_progress" in frames[1]
    assert '"status":"started"' in frames[1]
    assert "event: search_progress" in frames[2]
    assert '"result_count":3' in frames[2]


@pytest.mark.asyncio
async def test_thinking_events_require_explicit_capability_and_are_capped() -> None:
    event_bus.create_session_hub(
        "sess-thinking",
        thinking_enabled=True,
        thinking_supported=True,
        thinking_max_chars=5,
    )

    await event_bus.push_thinking_chunk(
        "sess-thinking",
        agent="verifier",
        stage="evaluation",
        text="123456789",
    )
    await event_bus.close_session_hub("sess-thinking")

    frames = await _collect_stream("sess-thinking", limit=5)
    assert len(frames) == 2
    assert '"text":"12345"' in frames[1]
    assert '"truncated":true' in frames[1]


@pytest.mark.asyncio
async def test_thinking_events_are_suppressed_by_default() -> None:
    event_bus.create_session_hub("sess-no-thinking")
    emitted = await event_bus.push_thinking_chunk(
        "sess-no-thinking",
        agent="verifier",
        stage="evaluation",
        text="private model trace",
    )

    assert emitted is False
    assert not event_bus.get_hub("sess-no-thinking").buffer


@pytest.mark.asyncio
async def test_retry_callback_publishes_safe_progress_message() -> None:
    event_bus.create_session_hub("sess-retry")
    with event_bus.event_scope("sess-retry", agent="verifier", stage="evaluation"):
        await event_bus.on_ollama_retry({"attempt": 2, "max_attempts": 3})

    stored = list(event_bus.get_hub("sess-retry").buffer)
    assert stored[0].event == "agent_progress"
    assert stored[0].data["message"] == "Model unavailable — retrying (2/3)"
    assert "exception" not in str(stored[0].data).lower()
