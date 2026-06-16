"""Fidelity gate for preventing truth-biased extractor rewrites."""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel, Field, field_validator

from factcheck.extractor.config import FIDELITY_CONFIG
from factcheck.extractor.prompts import FIDELITY_HUMAN_PROMPT, FIDELITY_SYSTEM_PROMPT
from factcheck.extractor.schemas import DisambiguatedContent, ExtractorState, PotentialClaim
from factcheck.extractor.utils.fidelity import (
    CoverageDecision,
    FidelityDecision,
    assess_claim_fidelity,
    assess_group_coverage,
)
from factcheck.llm.extractor_structured import call_extractor_structured_output
from factcheck.llm.factory import get_extractor_llm


logger = logging.getLogger(__name__)


class FidelityAuditOutput(BaseModel):
    """Structured output for the optional LLM fidelity audit."""

    faithful: bool
    reasoning: str = Field(default="", description="Brief explanation of the fidelity decision.")

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value


async def _audit_claim_fidelity(potential_claim: PotentialClaim) -> bool:
    """Use the extractor LLM only for borderline source/claim fidelity cases."""

    llm = get_extractor_llm(
        temperature=FIDELITY_CONFIG["temperature"],
        num_predict=FIDELITY_CONFIG["num_predict"],
        num_ctx=FIDELITY_CONFIG["num_ctx"],
    )
    response = await call_extractor_structured_output(
        llm=llm,
        output_class=FidelityAuditOutput,
        messages=[
            ("system", FIDELITY_SYSTEM_PROMPT),
            (
                "human",
                FIDELITY_HUMAN_PROMPT.format(
                    source_sentence=potential_claim.disambiguated_sentence,
                    original_sentence=potential_claim.original_sentence,
                    claim=potential_claim.claim_text,
                ),
            ),
        ],
        context_desc=f"fidelity audit for '{potential_claim.claim_text}'",
    )
    return bool(response and response.faithful)


def _with_status(claim: PotentialClaim, status: str) -> PotentialClaim:
    return claim.model_copy(update={"fidelity_status": status})


def _fallback_claim_from_potential(claim: PotentialClaim) -> PotentialClaim:
    logger.warning(
        "fidelity_fallback: using disambiguated sentence for original_index=%s",
        claim.original_index,
    )
    return PotentialClaim(
        claim_text=claim.disambiguated_sentence,
        disambiguated_sentence=claim.disambiguated_sentence,
        original_sentence=claim.original_sentence,
        original_index=claim.original_index,
        fidelity_status="fallback",
    )


def _fallback_claim_from_disambiguated(item: DisambiguatedContent) -> PotentialClaim:
    original = item.original_selected_item.original_context_item
    logger.warning(
        "fidelity_fallback: decomposition produced no faithful claims for original_index=%s",
        original.original_index,
    )
    return PotentialClaim(
        claim_text=item.disambiguated_sentence,
        disambiguated_sentence=item.disambiguated_sentence,
        original_sentence=original.original_sentence,
        original_index=original.original_index,
        fidelity_status="fallback",
    )


def _dedupe_claims(claims: list[PotentialClaim]) -> list[PotentialClaim]:
    deduped: list[PotentialClaim] = []
    seen: set[str] = set()
    for claim in claims:
        key = claim.claim_text.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


async def _faithful_claims_for_group(claims: list[PotentialClaim]) -> list[PotentialClaim]:
    faithful: list[PotentialClaim] = []
    for claim in claims:
        assessment = assess_claim_fidelity(
            claim_text=claim.claim_text,
            source_sentence=claim.disambiguated_sentence,
            context_text=claim.original_sentence,
        )
        if assessment.decision == FidelityDecision.PASS:
            faithful.append(_with_status(claim, "faithful"))
            continue
        if assessment.decision == FidelityDecision.BORDERLINE and await _audit_claim_fidelity(claim):
            faithful.append(_with_status(claim, "faithful"))
            continue
        logger.info(
            "Discarded unfaithful extracted claim: %s (source: %s; reason: %s; extra: %s)",
            claim.claim_text,
            claim.disambiguated_sentence,
            assessment.reason,
            sorted(assessment.extra_terms),
        )
    return faithful


async def fidelity_node(state: ExtractorState) -> dict[str, list[PotentialClaim]]:
    """Discard truth-corrected claims and fall back to the disambiguated source assertion."""

    if not state.potential_claims:
        return {
            "potential_claims": [
                _fallback_claim_from_disambiguated(item) for item in state.disambiguated_contents
            ]
        }

    grouped: dict[tuple[int, str], list[PotentialClaim]] = defaultdict(list)
    for claim in state.potential_claims:
        grouped[(claim.original_index, claim.disambiguated_sentence)].append(claim)

    output: list[PotentialClaim] = []
    for claims in grouped.values():
        faithful = await _faithful_claims_for_group(claims)
        if not faithful:
            output.append(_fallback_claim_from_potential(claims[0]))
            continue

        coverage = assess_group_coverage(
            source_sentence=claims[0].disambiguated_sentence,
            claim_texts=[claim.claim_text for claim in faithful],
        )
        if coverage.decision == CoverageDecision.INCOMPLETE:
            logger.warning(
                "decomposition_incomplete: original_index=%s — %s (uncovered: %s)",
                claims[0].original_index,
                coverage.reason,
                list(coverage.uncovered_segments),
            )
            output.append(_fallback_claim_from_potential(claims[0]))
            continue

        output.extend(faithful)

    return {"potential_claims": _dedupe_claims(output)}
