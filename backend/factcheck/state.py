"""Frozen after Phase 1 - bump version if changed."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from factcheck.extractor.schemas import ValidatedClaim


PipelineStatus = Literal["idle", "running", "done", "error"]
Verdict = Literal["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]


class ClaimResult(TypedDict):
    """Complete verification result for a single extracted claim."""

    claim: str
    verdict: Verdict
    confidence: float
    evidence: list[str]
    sources: list[str]
    reasoning: str
    search_queries: list[str]


class FactCheckState(TypedDict):
    """Shared state object passed through the LangGraph pipeline."""

    raw_input: str
    extracted_claims: list[ValidatedClaim]
    claim_results: list[ClaimResult]
    final_report: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    session_id: str
    error: str | None
    status: PipelineStatus
