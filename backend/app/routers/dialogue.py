"""FastAPI routes for the Dialogue Agent."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from factcheck.db.session_store import (
    load_session_for_dialogue,
    persist_dialogue_state,
    save_factcheck_session,
    session_exists,
)
from factcheck.dialogue import run_dialogue
from factcheck.graph.pipeline import run_factcheck_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dialogue", tags=["dialogue"])


class DialogueRequest(BaseModel):
    message: str = Field(min_length=1)


class DialogueResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    needs_new_factcheck: bool
    new_claim_text: str | None = None
    error: str | None = None


async def _trigger_new_factcheck(session_id: str, claim_text: str) -> None:
    """Run a new fact-check for a claim submitted during dialogue."""
    try:
        result = await run_factcheck_pipeline(session_id=session_id, text=claim_text)
        save_factcheck_session(
            session_id,
            raw_input=claim_text,
            claim_results=[dict(cr) for cr in result.get("claim_results", [])],
            final_report=result.get("final_report"),
        )
    except Exception as exc:
        logger.error(
            "[dialogue_route] New fact-check failed for session %s: %s",
            session_id,
            exc,
        )


@router.post("/{session_id}", response_model=DialogueResponse)
async def dialogue(session_id: str, request: DialogueRequest) -> DialogueResponse:
    """Handle a follow-up message for an existing fact-check session."""
    if not session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Complete a fact-check for this session first."
            ),
        )

    session = load_session_for_dialogue(session_id)
    prior_history_len = len(session["dialogue_history"])

    result = await run_dialogue(
        session_id=session_id,
        user_message=request.message,
        raw_input=session["raw_input"],
        claim_results=session["claim_results"],
        final_report=session.get("final_report"),
        dialogue_history=session["dialogue_history"],
        conversation_summary=session.get("conversation_summary"),
        compressed_fc_context=session.get("compressed_fc_context"),
    )

    persist_dialogue_state(
        session_id,
        result,
        prior_history_len=prior_history_len,
    )

    if result.get("needs_new_factcheck") and result.get("new_claim_text"):
        asyncio.create_task(
            _trigger_new_factcheck(session_id, result["new_claim_text"])
        )

    return DialogueResponse(
        session_id=session_id,
        response=result["response"],
        intent=result["intent"],
        needs_new_factcheck=result.get("needs_new_factcheck", False),
        new_claim_text=result.get("new_claim_text"),
        error=result.get("error"),
    )
