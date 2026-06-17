"""FastAPI routes for session lifecycle and SSE streaming."""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.sessions import (
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteResponse,
    PostMessageRequest,
    PostMessageResponse,
    SessionDetail,
    SessionSummary,
)
from factcheck.db.session_store import (
    complete_factcheck_run,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    save_user_message,
    session_exists,
    set_active_run,
    try_acquire_session,
    update_session_status,
)
from factcheck.dialogue.service import run_dialogue_turn_background
from factcheck.graph.event_bus import (
    StreamUnavailable,
    create_session_hub,
    resolve_stream,
    stream_events,
)
from factcheck.graph.runner import run_factcheck_with_events

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


async def _run_and_persist(session_id: str, run_id: str, text: str) -> None:
    """Background task: run the pipeline and persist final state."""
    try:
        result = await run_factcheck_with_events(session_id=session_id, text=text)
        claim_results = [dict(cr) for cr in result.get("claim_results", [])]
        await asyncio.to_thread(
            complete_factcheck_run,
            run_id,
            claim_results=claim_results,
            final_report=result.get("final_report"),
        )
        await asyncio.to_thread(set_active_run, session_id, run_id)
        await asyncio.to_thread(update_session_status, session_id, "done")
    except Exception as exc:
        logger.error(
            "[sessions] Pipeline failed for session %s: %s",
            session_id,
            exc,
        )
        await asyncio.to_thread(
            update_session_status,
            session_id,
            "error",
            error=str(exc),
        )


@router.post("", response_model=CreateSessionResponse, status_code=202)
async def start_session(
    body: CreateSessionRequest,
    background_tasks: BackgroundTasks,
) -> CreateSessionResponse:
    """Create a new fact-check session and kick off the pipeline in the background."""
    session_id = str(uuid.uuid4())

    run_id = await asyncio.to_thread(create_session, session_id, body.input)
    create_session_hub(session_id, run_id=run_id)
    background_tasks.add_task(_run_and_persist, session_id, run_id, body.input)

    return CreateSessionResponse(session_id=session_id, status="running")


@router.get("/{session_id}/stream")
async def stream_session(session_id: str) -> StreamingResponse:
    """SSE endpoint streaming pipeline progress for a session."""
    if not await asyncio.to_thread(session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session = await asyncio.to_thread(get_session, session_id)
    assert session is not None

    try:
        await resolve_stream(session_id, session)
    except StreamUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "session_status": exc.session_status,
                "active_run_id": exc.active_run_id,
                "hint": (
                    "Open the stream immediately after POST 202, or use "
                    f"GET /api/sessions/{session_id} for final state."
                ),
            },
        ) from exc

    return StreamingResponse(
        stream_events(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session_detail(session_id: str) -> SessionDetail:
    """Return full session state including claim results and messages."""
    session = await asyncio.to_thread(get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(**session)


@router.post("/{session_id}/messages", response_model=PostMessageResponse, status_code=202)
async def post_message(
    session_id: str,
    body: PostMessageRequest,
    background_tasks: BackgroundTasks,
) -> PostMessageResponse:
    """Post a follow-up message after the initial fact-check is complete."""
    if not await asyncio.to_thread(session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if not await asyncio.to_thread(try_acquire_session, session_id):
        raise HTTPException(status_code=409, detail="Session pipeline is not finished yet")

    message_id = str(
        await asyncio.to_thread(save_user_message, session_id, body.message)
    )
    create_session_hub(session_id)
    background_tasks.add_task(run_dialogue_turn_background, session_id, body.message)

    return PostMessageResponse(message_id=message_id)


@router.get("", response_model=list[SessionSummary])
async def list_all_sessions() -> list[SessionSummary]:
    """List all sessions ordered by newest first."""
    sessions = await asyncio.to_thread(list_sessions)
    return [SessionSummary(**session) for session in sessions]


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session_endpoint(session_id: str) -> DeleteResponse:
    """Delete a session and all related data."""
    if not await asyncio.to_thread(session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    deleted = await asyncio.to_thread(delete_session, session_id)
    return DeleteResponse(deleted=deleted)
