"""Evidence evaluator node for iterative claim verification."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.state import ClaimResult, Verdict
from factcheck.verifier.config import EVAL_NUM_CTX, EVALUATOR_TEMPERATURE
from factcheck.verifier.prompts import (
    EVIDENCE_EVALUATOR_HUMAN_PROMPT,
    EVIDENCE_EVALUATOR_REMINDER,
    EVIDENCE_EVALUATOR_SYSTEM_PROMPT,
)
from factcheck.verifier.schemas import CachedEvaluation, EvidenceItem, IntermediateAssessment, VerifierState
from factcheck.verifier.utils import format_evidence
from factcheck.verifier.utils.framing import extract_evaluation_frame

_VERDICT_CORRECTION_PROMPT = (
    "You set predicate_resolved_by_evidence=true and listed refuting_sources, "
    "but verdict was INSUFFICIENT_EVIDENCE. If authoritative sources refute the "
    "core predicate, verdict must be REFUTED. Do not require a study for the exact "
    "numeric qualifier."
)

_AUTHORITATIVE_TIERS = frozenset({"high", "medium"})


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
    core_predicate: str = ""
    predicate_resolved_by_evidence: bool = False
    refuting_sources: list[int] = Field(default_factory=list)
    needs_more_evidence: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
    influential_sources: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_predicate_resolution_consistency(self) -> Self:
        if not self.predicate_resolved_by_evidence:
            return self
        self.needs_more_evidence = False
        self.missing_aspects = []
        return self


def validated_refuting_sources(
    evidence: list[EvidenceItem],
    refuting_sources: list[int],
) -> list[int]:
    """Return 1-based indices that exist and are high/medium tier."""
    validated: list[int] = []
    for index in refuting_sources:
        if not (1 <= index <= len(evidence)):
            continue
        if evidence[index - 1].credibility_tier in _AUTHORITATIVE_TIERS:
            validated.append(index)
    return validated


def is_predicate_resolution_incoherent(
    response: EvaluationOutput,
    evidence: list[EvidenceItem],
) -> bool:
    """True when the model claims predicate resolution but verdict is insufficient."""
    return (
        response.verdict == "INSUFFICIENT_EVIDENCE"
        and response.predicate_resolved_by_evidence
        and len(validated_refuting_sources(evidence, response.refuting_sources)) >= 1
    )


def _coerced_refuted_reasoning(refuting_indices: list[int]) -> str:
    source_list = ", ".join(f"Source {index}" for index in refuting_indices[:3])
    return (
        f"Authoritative evidence ({source_list}) contradicts the claim's core predicate. "
        "General expert consensus refutes the asserted mechanism; an exact numeric "
        "threshold study is not required."
    )


def _coerce_incoherent_refutation(
    response: EvaluationOutput,
    evidence: list[EvidenceItem],
) -> EvaluationOutput:
    validated = validated_refuting_sources(evidence, response.refuting_sources)
    return response.model_copy(
        update={
            "verdict": "REFUTED",
            "needs_more_evidence": False,
            "missing_aspects": [],
            "influential_sources": validated[:5] or response.influential_sources,
            "reasoning": response.reasoning or _coerced_refuted_reasoning(validated),
        }
    )


def _apply_search_loop_policy(response: EvaluationOutput) -> EvaluationOutput:
    if not response.predicate_resolved_by_evidence:
        return response
    return response.model_copy(
        update={
            "needs_more_evidence": False,
            "missing_aspects": [],
        }
    )


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
        core_predicate=response.core_predicate,
        predicate_resolved_by_evidence=response.predicate_resolved_by_evidence,
        refuting_sources=response.refuting_sources,
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


async def _resolve_predicate_verdict(
    state: VerifierState,
    response: EvaluationOutput,
) -> EvaluationOutput:
    """Retry once on schema incoherence, then coerce if the model still contradicts itself."""
    if not is_predicate_resolution_incoherent(response, state.evidence):
        return response

    corrected = await _evaluate_with_llm(state, extra_human=_VERDICT_CORRECTION_PROMPT)
    if corrected is not None:
        response = corrected

    if is_predicate_resolution_incoherent(response, state.evidence):
        response = _coerce_incoherent_refutation(response, state.evidence)

    return response


async def evidence_evaluator_node(
    state: VerifierState,
) -> dict[str, ClaimResult | IntermediateAssessment | CachedEvaluation | int]:
    """Evaluate accumulated evidence and either finalize or request another search."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    response = await _evaluate_with_llm(state)

    if response is None:
        return _fallback_result(state)

    response = await _resolve_predicate_verdict(state, response)
    response = _apply_search_loop_policy(response)

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
