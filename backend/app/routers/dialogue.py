"""FastAPI routes for the Dialogue Agent."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from factcheck.db.session_store import try_acquire_session, update_session_status
from factcheck.dialogue.service import require_session, run_dialogue_turn

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


@router.post("/{session_id}", response_model=DialogueResponse)
async def dialogue(session_id: str, request: DialogueRequest) -> DialogueResponse:
    """Handle a follow-up message for an existing fact-check session."""
    try:
        require_session(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Complete a fact-check for this session first."
            ),
        ) from None

    if not await asyncio.to_thread(try_acquire_session, session_id):
        raise HTTPException(
            status_code=409,
            detail="Session pipeline is not finished yet",
        )

    result = None
    try:
        result = await run_dialogue_turn(session_id, request.message)
    except Exception:
        await asyncio.to_thread(
            update_session_status,
            session_id,
            "error",
        )
        raise
    else:
        return DialogueResponse(
            session_id=session_id,
            response=result["response"],
            intent=result["intent"],
            needs_new_factcheck=result.get("needs_new_factcheck", False),
            new_claim_text=result.get("new_claim_text"),
            error=result.get("error"),
        )
    finally:
        if result is not None and not result.get("needs_new_factcheck"):
            await asyncio.to_thread(update_session_status, session_id, "done")
