"""Validation node for final extractor claims."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field, field_validator

from factcheck.extractor.config import VALIDATION_CONFIG
from factcheck.extractor.utils.assertion_profile import looks_like_complete_declarative
from factcheck.extractor.prompts import VALIDATION_HUMAN_PROMPT, VALIDATION_SYSTEM_PROMPT
from factcheck.extractor.schemas import ExtractorState, PotentialClaim, ValidatedClaim
from factcheck.llm.extractor_structured import call_extractor_structured_output
from factcheck.llm.factory import get_extractor_llm


logger = logging.getLogger(__name__)


class ValidationOutput(BaseModel):
    """Structured output for claim validation."""

    is_complete_declarative: bool = Field(
        description="True if the claim is a complete, declarative sentence in isolation."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the validation decision.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value


async def _validate_claim(potential_claim: PotentialClaim) -> ValidatedClaim:
    llm = get_extractor_llm(
        temperature=VALIDATION_CONFIG["temperature"],
        num_predict=VALIDATION_CONFIG["num_predict"],
        num_ctx=VALIDATION_CONFIG["num_ctx"],
    )
    response = await call_extractor_structured_output(
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
        is_complete_declarative=bool(response and response.is_complete_declarative)
        or looks_like_complete_declarative(potential_claim.claim_text),
        disambiguated_sentence=potential_claim.disambiguated_sentence,
        original_sentence=potential_claim.original_sentence,
        original_index=potential_claim.original_index,
        fidelity_status=potential_claim.fidelity_status,
    )


async def validation_node(state: ExtractorState) -> dict[str, list[ValidatedClaim]]:
    """Keep complete, declarative, non-duplicate claims."""

    if not state.potential_claims:
        return {"validated_claims": []}

    # asyncio.gather schedules all coroutines concurrently, but the semaphore
    # inside call_extractor_structured_output (factcheck.llm.concurrency) means
    # at most OLLAMA_CONCURRENCY requests reach Ollama at any one time.
    validation_results = await asyncio.gather(
        *(_validate_claim(claim) for claim in state.potential_claims)
    )

    validated_claims: list[ValidatedClaim] = []
    seen_claims: set[str] = set()
    for validated in validation_results:
        if not validated.is_complete_declarative:
            logger.info("Discarded invalid claim: %s", validated.claim_text)
            continue
        normalized_key = validated.claim_text.strip().casefold()
        if normalized_key in seen_claims:
            logger.info("Discarded duplicate claim (case-insensitive): %s", validated.claim_text)
            continue
        seen_claims.add(normalized_key)
        validated_claims.append(validated)

    return {"validated_claims": validated_claims}