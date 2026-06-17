"""Shared dialogue turn execution for API routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from factcheck.db.session_store import (
    complete_factcheck_run,
    create_factcheck_run,
    invalidate_fc_context,
    load_session_for_dialogue,
    mark_factcheck_run_error,
    persist_dialogue_state,
    session_exists,
    set_active_run,
    try_acquire_session,
    update_session_status,
)
from factcheck.dialogue import run_dialogue
from factcheck.dialogue.schemas import DialogueOutput
from factcheck.graph.event_bus import create_session_hub
from factcheck.graph.runner import run_dialogue_with_events, run_factcheck_with_events

logger = logging.getLogger(__name__)


def _dialogue_load_kwargs(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_input": session["raw_input"],
        "claim_results": session["claim_results"],
        "final_report": session.get("final_report"),
        "dialogue_history": session["dialogue_history"],
        "conversation_summary": session.get("conversation_summary"),
        "compressed_fc_context": session.get("compressed_fc_context"),
        "fact_check_runs": session.get("fact_check_runs", []),
        "latest_run_sequence": session.get("latest_run_sequence", 0),
        "fc_context_covers_sequence": session.get("fc_context_covers_sequence"),
    }


async def run_dialogue_turn(session_id: str, message: str) -> DialogueOutput:
    """Run one synchronous dialogue turn and persist state."""
    session = load_session_for_dialogue(session_id)
    prior_history_len = len(session["dialogue_history"])

    result = await run_dialogue(
        session_id=session_id,
        user_message=message,
        **_dialogue_load_kwargs(session),
    )

    persist_dialogue_state(session_id, result, prior_history_len=prior_history_len)

    if result.get("needs_new_factcheck") and result.get("new_claim_text"):
        create_session_hub(session_id)
        asyncio.create_task(
            _run_new_factcheck_background(session_id, result["new_claim_text"])
        )

    return result


async def run_dialogue_turn_background(session_id: str, message: str) -> None:
    """Run a dialogue turn in the background with SSE events."""
    try:
        session = await asyncio.to_thread(load_session_for_dialogue, session_id)
        prior_history_len = len(session["dialogue_history"])

        result = await run_dialogue_with_events(
            session_id=session_id,
            user_message=message,
            **_dialogue_load_kwargs(session),
        )

        await asyncio.to_thread(
            persist_dialogue_state,
            session_id,
            result,
            prior_history_len=prior_history_len,
        )

        if result.get("needs_new_factcheck") and result.get("new_claim_text"):
            await _trigger_new_factcheck(
                session_id, result["new_claim_text"], lock_held=True
            )
        else:
            await asyncio.to_thread(update_session_status, session_id, "done")
    except Exception as exc:
        logger.error(
            "[dialogue_service] Background dialogue failed for session %s: %s",
            session_id,
            exc,
        )
        await asyncio.to_thread(
            update_session_status,
            session_id,
            "error",
            error=str(exc),
        )


async def _run_new_factcheck_background(session_id: str, claim_text: str) -> None:
    """Run a new fact-check in the background after a sync dialogue turn."""
    try:
        await _trigger_new_factcheck(session_id, claim_text, lock_held=True)
    except Exception:
        logger.exception(
            "[dialogue_service] Unhandled error in background fact-check for %s",
            session_id,
        )
        await asyncio.to_thread(
            update_session_status,
            session_id,
            "error",
            error="Fact-check failed",
        )


async def _trigger_new_factcheck(
    session_id: str,
    claim_text: str,
    *,
    lock_held: bool = False,
) -> None:
    """Run a new fact-check for a claim submitted during dialogue."""
    if not lock_held:
        acquired = await asyncio.to_thread(try_acquire_session, session_id)
        if not acquired:
            logger.warning(
                "[dialogue_service] Re-factcheck skipped for session %s: session busy",
                session_id,
            )
            return

    run_id = await asyncio.to_thread(
        create_factcheck_run,
        session_id,
        claim_text,
        "dialogue",
    )
    create_session_hub(session_id, run_id=run_id)
    try:
        result = await run_factcheck_with_events(
            session_id=session_id,
            text=claim_text,
            extraction_mode="claim",
        )
        claim_results = [dict(cr) for cr in result.get("claim_results", [])]
        await asyncio.to_thread(
            complete_factcheck_run,
            run_id,
            claim_results=claim_results,
            final_report=result.get("final_report"),
        )
        await asyncio.to_thread(set_active_run, session_id, run_id)
        await asyncio.to_thread(invalidate_fc_context, session_id)
        await asyncio.to_thread(update_session_status, session_id, "done")
    except Exception as exc:
        logger.error(
            "[dialogue_service] New fact-check failed for session %s: %s",
            session_id,
            exc,
        )
        await asyncio.to_thread(mark_factcheck_run_error, run_id, str(exc))
        await asyncio.to_thread(
            update_session_status,
            session_id,
            "error",
            error=str(exc),
        )


def require_session(session_id: str) -> None:
    """Raise KeyError if the session does not exist."""
    if not session_exists(session_id):
        raise KeyError(session_id)
