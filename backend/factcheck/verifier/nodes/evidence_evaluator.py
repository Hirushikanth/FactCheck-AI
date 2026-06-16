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
    EVIDENCE_EVALUATOR_REMINDER,
    EVIDENCE_EVALUATOR_SYSTEM_PROMPT,
)
from factcheck.verifier.schemas import CachedEvaluation, IntermediateAssessment, VerifierState
from factcheck.verifier.utils import format_evidence
from factcheck.verifier.utils.framing import extract_evaluation_frame
from factcheck.verifier.utils.verdict_signals import (
    count_authoritative_contradictions,
    reasoning_suggests_contradiction,
)

_MIN_CONTRADICTIONS_FOR_REFUTED = 2
_VERDICT_CORRECTION_PROMPT = (
    "Your reasoning suggests contradictory evidence, but verdict was INSUFFICIENT_EVIDENCE. "
    "Re-evaluate: if authoritative sources contradict the claim's core predicate, "
    "verdict must be REFUTED."
)


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
        "source_sentence": state.source_sentence,
        "fidelity_status": state.fidelity_status,
    }


def _to_cached(response: EvaluationOutput) -> CachedEvaluation:
    return CachedEvaluation(
        verdict=response.verdict,
        confidence=response.confidence,
        reasoning=response.reasoning,
        needs_more_evidence=response.needs_more_evidence,
        missing_aspects=response.missing_aspects,
        influential_sources=response.influential_sources,
    )


def _fallback_result(
    state: VerifierState,
) -> dict[str, ClaimResult | IntermediateAssessment | CachedEvaluation]:
    if state.cached_evaluation is not None:
        cached = state.cached_evaluation
        _mark_influential_sources(state, cached.influential_sources)
        return {
            "claim_result": _claim_result(
                state,
                verdict=cached.verdict,
                confidence=cached.confidence,
                reasoning=cached.reasoning,
            ),
            "intermediate_assessment": IntermediateAssessment(
                needs_more_evidence=False,
                missing_aspects=cached.missing_aspects,
            ),
        }

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


def _evaluation_messages(
    state: VerifierState,
    *,
    extra_human: str | None = None,
) -> list[tuple[str, str]]:
    source_sentence = state.source_sentence or state.claim_text
    evaluation_frame = extract_evaluation_frame(state.claim_text)
    frame_block = (
        f"Evaluation frame:\n{evaluation_frame}\n\n" if evaluation_frame else ""
    )
    human_content = (
        frame_block
        + EVIDENCE_EVALUATOR_HUMAN_PROMPT.format(
            source_sentence=source_sentence,
            claim=state.claim_text,
            evidence=format_evidence(state.evidence),
        )
        + EVIDENCE_EVALUATOR_REMINDER
    )
    if extra_human:
        human_content += f"\n{extra_human}"

    return [
        ("system", EVIDENCE_EVALUATOR_SYSTEM_PROMPT),
        ("human", human_content),
    ]


def _guardrail_refuted_reasoning(contradicting_indices: list[int]) -> str:
    source_list = ", ".join(f"Source {index}" for index in contradicting_indices[:3])
    return (
        f"Authoritative evidence ({source_list}) contradicts the claim's core predicate. "
        "General expert consensus refutes the asserted mechanism; an exact numeric "
        "threshold study is not required."
    )


def _apply_verdict_guardrails(
    state: VerifierState,
    response: EvaluationOutput,
) -> EvaluationOutput:
    """Correct clear model mistakes when authoritative evidence already refutes the claim."""
    if response.verdict not in ("INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"):
        return response

    contradicting_indices = count_authoritative_contradictions(
        claim_text=state.claim_text,
        evidence=state.evidence,
        influential_indices=response.influential_sources,
    )
    if len(contradicting_indices) < _MIN_CONTRADICTIONS_FOR_REFUTED:
        return response

    return response.model_copy(
        update={
            "verdict": "REFUTED",
            "confidence": max(response.confidence, 0.75),
            "reasoning": _guardrail_refuted_reasoning(contradicting_indices),
            "needs_more_evidence": False,
            "missing_aspects": [],
            "influential_sources": contradicting_indices[:5] or response.influential_sources,
        }
    )


def _should_request_verdict_correction(
    state: VerifierState,
    response: EvaluationOutput,
) -> bool:
    if response.verdict != "INSUFFICIENT_EVIDENCE":
        return False
    if not reasoning_suggests_contradiction(response.reasoning):
        return False
    contradicting_indices = count_authoritative_contradictions(
        claim_text=state.claim_text,
        evidence=state.evidence,
        influential_indices=response.influential_sources,
    )
    return len(contradicting_indices) >= 1


async def _evaluate_with_llm(
    state: VerifierState,
    *,
    extra_human: str | None = None,
) -> EvaluationOutput | None:
    llm = get_verifier_llm(temperature=EVALUATOR_TEMPERATURE, num_ctx=EVAL_NUM_CTX)
    return await call_llm_with_structured_output(
        llm=llm,
        output_class=EvaluationOutput,
        messages=_evaluation_messages(state, extra_human=extra_human),
        context_desc=f"evidence evaluation for '{state.claim_text}'",
    )


async def evidence_evaluator_node(
    state: VerifierState,
) -> dict[str, ClaimResult | IntermediateAssessment | CachedEvaluation | int]:
    """Evaluate accumulated evidence and either finalize or request another search."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    response = await _evaluate_with_llm(state)

    if response is None:
        return _fallback_result(state)

    if _should_request_verdict_correction(state, response):
        corrected = await _evaluate_with_llm(state, extra_human=_VERDICT_CORRECTION_PROMPT)
        if corrected is not None:
            response = corrected

    response = _apply_verdict_guardrails(state, response)

    _mark_influential_sources(state, response.influential_sources)
    intermediate = IntermediateAssessment(
        needs_more_evidence=response.needs_more_evidence,
        missing_aspects=response.missing_aspects,
    )
    cached = _to_cached(response)
    should_retry = (
        not state.search_exhausted
        and response.verdict == "INSUFFICIENT_EVIDENCE"
        and response.needs_more_evidence
        and state.iteration_count + 1 < state.max_iterations
    )

    if should_retry:
        return {
            "intermediate_assessment": intermediate,
            "iteration_count": state.iteration_count + 1,
            "cached_evaluation": cached,
        }

    return {
        "claim_result": _claim_result(
            state,
            verdict=response.verdict,
            confidence=response.confidence,
            reasoning=response.reasoning,
        ),
        "intermediate_assessment": intermediate,
        "cached_evaluation": cached,
    }
