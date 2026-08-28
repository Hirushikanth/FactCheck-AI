"""Shared ClaimResult construction for verifier nodes and agent."""

from __future__ import annotations

from factcheck.state import ClaimResult, ProcessingStatus, Verdict
from factcheck.verifier.config import (
    MAX_DIALOGUE_EVIDENCE_ITEMS,
    MAX_DIALOGUE_EVIDENCE_WORDS,
)
from factcheck.verifier.schemas import VerifierState
from factcheck.verifier.utils import truncate_snippet


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


def _dialogue_evidence(state: VerifierState) -> list[str]:
    """Select and word-cap the verifier evidence safe for dialogue context."""
    influential = [item for item in state.evidence if item.is_influential]
    selected = (influential or state.evidence)[:MAX_DIALOGUE_EVIDENCE_ITEMS]
    return [
        truncate_snippet(item.snippet, max_words=MAX_DIALOGUE_EVIDENCE_WORDS)
        for item in selected
    ]


def build_claim_result(
    *,
    claim: str,
    verdict: Verdict,
    confidence: float,
    reasoning: str,
    evidence: list[str] | None = None,
    sources: list[str] | None = None,
    search_queries: list[str] | None = None,
    source_sentence: str | None = None,
    fidelity_status: str | None = None,
    processing_status: ProcessingStatus = "ok",
    processing_error: str | None = None,
) -> ClaimResult:
    """Build a ClaimResult dict, omitting processing fields on the happy path."""
    result: ClaimResult = {
        "claim": claim,
        "verdict": verdict,
        "confidence": _clamp_confidence(confidence),
        "evidence": evidence or [],
        "sources": sources or [],
        "reasoning": reasoning,
        "search_queries": search_queries or [],
        "source_sentence": source_sentence,
        "fidelity_status": fidelity_status,
    }
    if processing_status != "ok":
        result["processing_status"] = processing_status
        if processing_error is not None:
            result["processing_error"] = processing_error
    return result


def build_claim_result_from_state(
    state: VerifierState,
    *,
    verdict: Verdict,
    confidence: float,
    reasoning: str,
    processing_status: ProcessingStatus = "ok",
    processing_error: str | None = None,
) -> ClaimResult:
    """Build a ClaimResult from verifier subgraph state."""
    result = build_claim_result(
        claim=state.claim_text,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        evidence=[item.snippet for item in state.evidence],
        sources=[item.url for item in state.evidence],
        search_queries=state.all_queries,
        source_sentence=state.source_sentence,
        fidelity_status=state.fidelity_status,
        processing_status=processing_status,
        processing_error=processing_error,
    )
    result["dialogue_evidence"] = _dialogue_evidence(state)
    return result


def is_processing_error(result: ClaimResult) -> bool:
    """True when verification failed due to a system error, not a semantic verdict."""
    return result.get("processing_status") == "error"
