"""Verdict engine node for evidence-grounded claim verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.state import ClaimResult, Verdict
from factcheck.verifier.config import VERDICT_TEMPERATURE
from factcheck.verifier.prompts import VERDICT_HUMAN_PROMPT, VERDICT_SYSTEM_PROMPT
from factcheck.verifier.schemas import EvidenceItem, VerifierState


class VerdictOutput(BaseModel):
    """Structured output for the verdict engine."""

    verdict: Literal["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    return "\n\n".join(
        (
            f"{index}. Source: {item.url}\n"
            f"Title: {item.title or 'Untitled'}\n"
            f"Snippet: {item.snippet}"
        )
        for index, item in enumerate(evidence, start=1)
    )


def _claim_result(
    state: VerifierState,
    *,
    verdict: Verdict,
    confidence: float,
    reasoning: str,
) -> ClaimResult:
    return {
        "claim": state.claim,
        "verdict": verdict,
        "confidence": _clamp_confidence(confidence),
        "evidence": [item.snippet for item in state.ranked_evidence],
        "sources": [item.url for item in state.ranked_evidence],
        "reasoning": reasoning,
        "search_queries": state.search_queries,
    }


async def verdict_engine_node(state: VerifierState) -> dict[str, ClaimResult]:
    """Produce a final verdict from ranked evidence snippets."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    if not state.ranked_evidence:
        return {
            "claim_result": _claim_result(
                state,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning="No relevant evidence was available to verify this claim.",
            )
        }

    llm = get_verifier_llm(temperature=VERDICT_TEMPERATURE)
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=VerdictOutput,
        messages=[
            ("system", VERDICT_SYSTEM_PROMPT),
            (
                "human",
                VERDICT_HUMAN_PROMPT.format(
                    claim=state.claim,
                    evidence=_format_evidence(state.ranked_evidence),
                ),
            ),
        ],
        context_desc=f"verdict for '{state.claim}'",
    )

    if response is None:
        return {
            "claim_result": _claim_result(
                state,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning="The verifier could not produce a structured verdict from the evidence.",
            )
        }

    return {
        "claim_result": _claim_result(
            state,
            verdict=response.verdict,
            confidence=response.confidence,
            reasoning=response.reasoning,
        )
    }
