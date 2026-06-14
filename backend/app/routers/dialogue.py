"""FastAPI routes for the Dialogue Agent."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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

    result = await run_dialogue_turn(session_id, request.message)

    return DialogueResponse(
        session_id=session_id,
        response=result["response"],
        intent=result["intent"],
        needs_new_factcheck=result.get("needs_new_factcheck", False),
        new_claim_text=result.get("new_claim_text"),
        error=result.get("error"),
    )
