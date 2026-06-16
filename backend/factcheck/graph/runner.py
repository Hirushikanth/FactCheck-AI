"""Pipeline and dialogue runners with SSE event emission."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from factcheck.dialogue.schemas import ConversationSummary, DialogueOutput, DialogueTurn
from factcheck.graph.event_bus import close_session_queue, push_event
from factcheck.graph.pipeline import _initial_state, build_graph
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
    state: FactCheckState = _initial_state(session_id=session_id, text=text)
    seen_agents: set[str] = set()

    try:
        graph = build_graph()
        async for chunk in graph.astream(state, stream_mode="updates"):
            for node_name, update in chunk.items():
                current_agent = node_name
                state = {**state, **update}

                if node_name in _PIPELINE_AGENTS and node_name not in seen_agents:
                    seen_agents.add(node_name)
                    await push_event(
                        session_id,
                        "agent_start",
                        {"agent": node_name, "timestamp": _now_iso()},
                    )

                if node_name == "extractor":
                    claims = update.get("extracted_claims", [])
                    for index, claim_obj in enumerate(claims):
                        claim_text = getattr(claim_obj, "claim_text", str(claim_obj))
                        await push_event(
                            session_id,
                            "claim_found",
                            {
                                "claim": claim_text,
                                "index": index,
                                "total": len(claims),
                            },
                        )

                if node_name == "reporter" and update.get("final_report"):
                    await push_event(
                        session_id,
                        "report_ready",
                        {"final_report": update["final_report"]},
                    )

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
    fact_check_runs: Optional[list[dict[str, Any]]] = None,
    latest_run_sequence: int = 0,
    fc_context_covers_sequence: Optional[int] = None,
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
            fact_check_runs=fact_check_runs,
            latest_run_sequence=latest_run_sequence,
            fc_context_covers_sequence=fc_context_covers_sequence,
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
