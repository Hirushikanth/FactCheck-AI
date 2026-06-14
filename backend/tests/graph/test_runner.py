"""Tests for pipeline runner SSE emission."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.graph import event_bus, runner


@pytest.fixture(autouse=True)
def clear_queues():
    event_bus._queues.clear()
    yield
    event_bus._queues.clear()


def _validated_claim() -> ValidatedClaim:
    return ValidatedClaim(
        claim_text="Carrots improve eyesight.",
        is_complete_declarative=True,
        disambiguated_sentence="Carrots improve eyesight.",
        original_sentence="Carrots improve eyesight.",
        original_index=0,
    )


@pytest.mark.asyncio
async def test_run_factcheck_with_events_emits_contract_events(monkeypatch) -> None:
    claim = _validated_claim()
    final_state = {
        "raw_input": "Carrots improve eyesight.",
        "extracted_claims": [claim],
        "claim_results": [
            {
                "claim": "Carrots improve eyesight.",
                "verdict": "REFUTED",
                "confidence": 0.8,
                "evidence": ["No evidence."],
                "sources": ["https://example.com"],
                "reasoning": "Refuted.",
                "search_queries": ["carrots eyesight"],
            }
        ],
        "final_report": "# Report",
        "messages": [],
        "current_agent": "reporter",
        "session_id": "sess-runner",
        "error": None,
        "status": "done",
    }

    monkeypatch.setattr(
        runner,
        "run_factcheck_pipeline",
        AsyncMock(return_value=final_state),
    )

    event_bus.create_session_queue("sess-runner")
    collected: list[dict] = []

    async def capture_push(session_id: str, event: str, data: dict) -> None:
        collected.append({"event": event, "data": data})
        await event_bus.push_event(session_id, event, data)

    monkeypatch.setattr(runner, "push_event", capture_push)

    result = await runner.run_factcheck_with_events(
        session_id="sess-runner",
        text="Carrots improve eyesight.",
        started_at=0.0,
    )

    assert result["final_report"] == "# Report"
    event_names = [item["event"] for item in collected]
    assert event_names.count("agent_start") == 3
    assert "claim_found" in event_names
    assert "verdict_ready" in event_names
    assert "report_ready" in event_names
    assert event_names[-1] == "pipeline_done"


@pytest.mark.asyncio
async def test_run_factcheck_with_events_emits_pipeline_error(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "run_factcheck_pipeline",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    event_bus.create_session_queue("sess-error")
    collected: list[str] = []

    async def capture_push(session_id: str, event: str, data: dict) -> None:
        collected.append(event)
        await event_bus.push_event(session_id, event, data)

    monkeypatch.setattr(runner, "push_event", capture_push)

    with pytest.raises(RuntimeError, match="boom"):
        await runner.run_factcheck_with_events(session_id="sess-error", text="test")

    assert "pipeline_error" in collected
