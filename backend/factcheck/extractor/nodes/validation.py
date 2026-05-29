"""Validation node for final extractor claims."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from factcheck.extractor.config import VALIDATION_CONFIG
from factcheck.extractor.prompts import VALIDATION_HUMAN_PROMPT, VALIDATION_SYSTEM_PROMPT
from factcheck.extractor.schemas import ExtractorState, PotentialClaim, ValidatedClaim
from factcheck.llm.factory import get_extractor_llm
from factcheck.llm.structured import call_llm_with_structured_output


logger = logging.getLogger(__name__)


class ValidationOutput(BaseModel):
    """Structured output for claim validation."""

    is_complete_declarative: bool


async def _validate_claim(potential_claim: PotentialClaim) -> ValidatedClaim:
    llm = get_extractor_llm(temperature=VALIDATION_CONFIG["temperature"])
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=ValidationOutput,
        messages=[
            ("system", VALIDATION_SYSTEM_PROMPT),
            ("human", VALIDATION_HUMAN_PROMPT.format(claim=potential_claim.claim_text)),
        ],
        context_desc=f"validation for '{potential_claim.claim_text}'",
    )
    return ValidatedClaim(
        claim_text=potential_claim.claim_text,
        is_complete_declarative=bool(response and response.is_complete_declarative),
        disambiguated_sentence=potential_claim.disambiguated_sentence,
        original_sentence=potential_claim.original_sentence,
        original_index=potential_claim.original_index,
    )


async def validation_node(state: ExtractorState) -> dict[str, list[ValidatedClaim]]:
    """Keep complete, declarative, non-duplicate claims."""

    if not state.potential_claims:
        return {"validated_claims": []}

    validation_results = await asyncio.gather(
        *(_validate_claim(claim) for claim in state.potential_claims)
    )

    validated_claims: list[ValidatedClaim] = []
    seen_claims: set[str] = set()
    for validated in validation_results:
        if not validated.is_complete_declarative:
            logger.info("Discarded invalid claim: %s", validated.claim_text)
            continue
        if validated.claim_text in seen_claims:
            logger.info("Discarded duplicate claim: %s", validated.claim_text)
            continue
        seen_claims.add(validated.claim_text)
        validated_claims.append(validated)

    return {"validated_claims": validated_claims}
