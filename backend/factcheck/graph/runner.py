"""Pipeline and dialogue runners with SSE event emission."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from factcheck.dialogue.schemas import ConversationSummary, DialogueOutput, DialogueTurn
from factcheck.graph.event_bus import close_session_queue, push_event
from factcheck.graph.pipeline import run_factcheck_pipeline
from factcheck.state import FactCheckState

_PIPELINE_AGENTS = ("extractor", "verifier", "reporter")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_factcheck_with_events(
    *,
    session_id: str,
    text: str,
    started_at: float | None = None,
) -> FactCheckState:
    """Run the fact-check pipeline and emit SSE events for each stage."""
    start = started_at if started_at is not None else time.monotonic()
    current_agent = "extractor"

    try:
        for agent in _PIPELINE_AGENTS:
            current_agent = agent
            await push_event(
                session_id,
                "agent_start",
                {"agent": agent, "timestamp": _now_iso()},
            )

        state = await run_factcheck_pipeline(session_id=session_id, text=text)

        claims = state.get("extracted_claims", [])
        for index, claim_obj in enumerate(claims):
            claim_text = getattr(claim_obj, "claim_text", str(claim_obj))
            await push_event(
                session_id,
                "claim_found",
                {"claim": claim_text, "index": index, "total": len(claims)},
            )

        for result in state.get("claim_results", []):
            await push_event(
                session_id,
                "verdict_ready",
                {
                    "claim": result["claim"],
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                },
            )

        final_report = state.get("final_report")
        if final_report:
            await push_event(session_id, "report_ready", {"final_report": final_report})

        duration = time.monotonic() - start
        await push_event(
            session_id,
            "pipeline_done",
            {"session_id": session_id, "duration_seconds": round(duration, 2)},
        )
        return state
    except Exception as exc:
        await push_event(
            session_id,
            "pipeline_error",
            {"error": str(exc), "agent": current_agent},
        )
        raise
    finally:
        await close_session_queue(session_id)


async def run_dialogue_with_events(
    *,
    session_id: str,
    user_message: str,
    raw_input: str,
    claim_results: list[dict[str, Any]],
    final_report: Optional[str] = None,
    dialogue_history: Optional[list[DialogueTurn]] = None,
    conversation_summary: Optional[ConversationSummary] = None,
    compressed_fc_context: Optional[str] = None,
) -> DialogueOutput:
    """Run one dialogue turn and emit SSE events."""
    from factcheck.dialogue import run_dialogue

    try:
        await push_event(
            session_id,
            "agent_start",
            {"agent": "dialogue", "timestamp": _now_iso()},
        )

        result = await run_dialogue(
            session_id=session_id,
            user_message=user_message,
            raw_input=raw_input,
            claim_results=claim_results,
            final_report=final_report,
            dialogue_history=dialogue_history,
            conversation_summary=conversation_summary,
            compressed_fc_context=compressed_fc_context,
        )

        if result.get("response"):
            await push_event(
                session_id,
                "dialogue_reply",
                {"message": result["response"]},
            )

        await push_event(
            session_id,
            "pipeline_done",
            {"session_id": session_id, "duration_seconds": 0.0},
        )
        return result
    except Exception as exc:
        await push_event(
            session_id,
            "pipeline_error",
            {"error": str(exc), "agent": "dialogue"},
        )
        raise
    finally:
        await close_session_queue(session_id)
