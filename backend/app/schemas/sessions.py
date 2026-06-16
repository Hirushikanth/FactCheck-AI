"""Pydantic models for session API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    input: str = Field(min_length=1)


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str


class PostMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class PostMessageResponse(BaseModel):
    message_id: str


class SessionSummary(BaseModel):
    session_id: str
    raw_input: str
    status: str
    created_at: float
    updated_at: float


class FactCheckRunSummary(BaseModel):
    run_id: str
    sequence: int
    raw_input: str
    status: str
    triggered_by: str
    created_at: float


class SessionDetail(BaseModel):
    session_id: str
    active_run_id: str | None = None
    raw_input: str
    status: str
    final_report: str | None
    error: str | None
    claim_results: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    runs: list[FactCheckRunSummary] = Field(default_factory=list)
    created_at: float
    updated_at: float


class DeleteResponse(BaseModel):
    deleted: bool
