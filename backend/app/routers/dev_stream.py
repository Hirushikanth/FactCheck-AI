"""Development-only SSE routes for LangGraph streams."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from factcheck.streaming.extractor_runner import stream_extractor_updates
from factcheck.streaming.sse import format_sse


router = APIRouter(prefix="/api/dev", tags=["dev-stream"])


class ExtractorStreamRequest(BaseModel):
    """Request body for the temporary extractor stream endpoint."""

    input: str = Field(min_length=1)
    metadata: str | None = None


@router.post("/extractor/stream")
async def stream_extractor(request: ExtractorStreamRequest) -> StreamingResponse:
    """Stream extractor graph updates as Server-Sent Events."""

    async def event_stream() -> AsyncIterator[str]:
        async for payload in stream_extractor_updates(request.input, request.metadata):
            yield format_sse(payload["event"], payload["data"])

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
