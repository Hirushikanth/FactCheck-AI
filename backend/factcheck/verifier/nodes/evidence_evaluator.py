"""Evidence evaluator node for iterative claim verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.state import ClaimResult, Verdict
from factcheck.verifier.config import EVAL_NUM_CTX, EVALUATOR_TEMPERATURE
from factcheck.verifier.prompts import (
    EVIDENCE_EVALUATOR_HUMAN_PROMPT,
    EVIDENCE_EVALUATOR_SYSTEM_PROMPT,
)
from factcheck.verifier.schemas import IntermediateAssessment, VerifierState
from factcheck.verifier.utils import format_evidence


class EvaluationOutput(BaseModel):
    """Structured output for evidence evaluation and retry control."""

    verdict: Literal[
        "SUPPORTED",
        "REFUTED",
        "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    needs_more_evidence: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
    influential_sources: list[int] = Field(default_factory=list)


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


def _mark_influential_sources(state: VerifierState, source_indices: list[int]) -> None:
    influential = {index for index in source_indices if 1 <= index <= len(state.evidence)}
    for index, item in enumerate(state.evidence, start=1):
        item.is_influential = index in influential


def _claim_result(
    state: VerifierState,
    *,
    verdict: Verdict,
    confidence: float,
    reasoning: str,
) -> ClaimResult:
    return {
        "claim": state.claim_text,
        "verdict": verdict,
        "confidence": _clamp_confidence(confidence),
        "evidence": [item.snippet for item in state.evidence],
        "sources": [item.url for item in state.evidence],
        "reasoning": reasoning,
        "search_queries": state.all_queries,
    }


def _fallback_result(state: VerifierState) -> dict[str, ClaimResult | IntermediateAssessment]:
    return {
        "claim_result": _claim_result(
            state,
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning="The verifier could not produce a structured verdict from the evidence.",
        ),
        "intermediate_assessment": IntermediateAssessment(
            needs_more_evidence=False,
            missing_aspects=[],
        ),
    }


async def evidence_evaluator_node(
    state: VerifierState,
) -> dict[str, ClaimResult | IntermediateAssessment | int]:
    """Evaluate accumulated evidence and either finalize or request another search."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    llm = get_verifier_llm(temperature=EVALUATOR_TEMPERATURE, num_ctx=EVAL_NUM_CTX)
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=EvaluationOutput,
        messages=[
            ("system", EVIDENCE_EVALUATOR_SYSTEM_PROMPT),
            (
                "human",
                EVIDENCE_EVALUATOR_HUMAN_PROMPT.format(
                    claim=state.claim_text,
                    evidence=format_evidence(state.evidence),
                ),
            ),
        ],
        context_desc=f"evidence evaluation for '{state.claim_text}'",
    )

    if response is None:
        return _fallback_result(state)

    _mark_influential_sources(state, response.influential_sources)
    intermediate = IntermediateAssessment(
        needs_more_evidence=response.needs_more_evidence,
        missing_aspects=response.missing_aspects,
    )
    should_retry = (
        response.verdict == "INSUFFICIENT_EVIDENCE"
        and response.needs_more_evidence
        and state.iteration_count + 1 < state.max_iterations
    )

    if should_retry:
        return {
            "intermediate_assessment": intermediate,
            "iteration_count": state.iteration_count + 1,
        }

    return {
        "claim_result": _claim_result(
            state,
            verdict=response.verdict,
            confidence=response.confidence,
            reasoning=response.reasoning,
        ),
        "intermediate_assessment": intermediate,
    }
