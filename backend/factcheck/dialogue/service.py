"""Shared dialogue turn execution for API routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from factcheck.db.session_store import (
    load_session_for_dialogue,
    persist_dialogue_state,
    save_factcheck_session,
    session_exists,
    update_session_status,
)
from factcheck.dialogue import run_dialogue
from factcheck.dialogue.schemas import DialogueOutput
from factcheck.graph.event_bus import create_session_queue
from factcheck.graph.runner import run_dialogue_with_events, run_factcheck_with_events

logger = logging.getLogger(__name__)


async def run_dialogue_turn(session_id: str, message: str) -> DialogueOutput:
    """Run one synchronous dialogue turn and persist state."""
    session = load_session_for_dialogue(session_id)
    prior_history_len = len(session["dialogue_history"])

    result = await run_dialogue(
        session_id=session_id,
        user_message=message,
        raw_input=session["raw_input"],
        claim_results=session["claim_results"],
        final_report=session.get("final_report"),
        dialogue_history=session["dialogue_history"],
        conversation_summary=session.get("conversation_summary"),
        compressed_fc_context=session.get("compressed_fc_context"),
    )

    persist_dialogue_state(session_id, result, prior_history_len=prior_history_len)

    if result.get("needs_new_factcheck") and result.get("new_claim_text"):
        asyncio.create_task(
            _trigger_new_factcheck(session_id, result["new_claim_text"])
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
            raw_input=session["raw_input"],
            claim_results=session["claim_results"],
            final_report=session.get("final_report"),
            dialogue_history=session["dialogue_history"],
            conversation_summary=session.get("conversation_summary"),
            compressed_fc_context=session.get("compressed_fc_context"),
        )

        await asyncio.to_thread(
            persist_dialogue_state,
            session_id,
            result,
            prior_history_len=prior_history_len,
        )

        if result.get("needs_new_factcheck") and result.get("new_claim_text"):
            await _trigger_new_factcheck(session_id, result["new_claim_text"])
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


async def _trigger_new_factcheck(session_id: str, claim_text: str) -> None:
    """Run a new fact-check for a claim submitted during dialogue."""
    create_session_queue(session_id)
    try:
        result = await run_factcheck_with_events(session_id=session_id, text=claim_text)
        claim_results = [dict(cr) for cr in result.get("claim_results", [])]
        await asyncio.to_thread(
            save_factcheck_session,
            session_id,
            raw_input=claim_text,
            claim_results=claim_results,
            final_report=result.get("final_report"),
        )
    except Exception as exc:
        logger.error(
            "[dialogue_service] New fact-check failed for session %s: %s",
            session_id,
            exc,
        )
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
