"""Decomposition node for atomic claim extraction."""

from __future__ import annotations

import asyncio
import itertools
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

from factcheck.extractor.config import DECOMPOSITION_CONFIG
from factcheck.extractor.prompts import DECOMPOSITION_SYSTEM_PROMPT, HUMAN_PROMPT
from factcheck.extractor.schemas import (
    DisambiguatedContent,
    ExtractorStageFailure,
    ExtractorState,
    PotentialClaim,
)
from factcheck.llm.extractor_structured import call_extractor_structured_output
from factcheck.llm.factory import get_extractor_llm


logger = logging.getLogger(__name__)


class DecompositionOutput(BaseModel):
    """Structured output for the decomposition stage."""

    no_claims: bool
    claims: list[str] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition decision.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> DecompositionOutput:
        if self.no_claims:
            return self
        if not self.claims:
            raise ValueError("claims must be non-empty when no_claims is false")
        return self


async def _decomposition_stage(
    item: DisambiguatedContent,
    llm: object,
) -> tuple[list[PotentialClaim], ExtractorStageFailure | None]:
    sentence = item.disambiguated_sentence
    context = item.original_selected_item.preceding_context_item.context_for_llm

    response = await call_extractor_structured_output(
        llm=llm,
        output_class=DecompositionOutput,
        messages=[
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT.format(excerpt=context, sentence=sentence)),
        ],
        context_desc=f"decomposition for '{sentence}'",
    )

    if not response:
        logger.warning("Decomposition parse failed for sentence: '%s'", sentence)
        return [], ExtractorStageFailure(
            stage="decomposition",
            sentence=sentence,
            reason="parse_failed",
            successes=0,
            attempts=1,
        )

    if response.no_claims or not response.claims:
        return [], None

    original = item.original_selected_item.original_context_item
    claims = [claim.strip() for claim in response.claims if claim.strip()]
    if not claims:
        return [], ExtractorStageFailure(
            stage="decomposition",
            sentence=sentence,
            reason="no_output",
            successes=0,
            attempts=1,
        )

    return [
        PotentialClaim(
            claim_text=claim,
            disambiguated_sentence=sentence,
            original_sentence=original.original_sentence,
            original_index=original.original_index,
        )
        for claim in claims
    ], None


async def decomposition_node(state: ExtractorState) -> dict[str, list[PotentialClaim] | list]:
    """Break disambiguated sentences into independently verifiable claims."""

    if not state.disambiguated_contents:
        return {"potential_claims": []}

    llm = get_extractor_llm(
        temperature=DECOMPOSITION_CONFIG["temperature"],
        num_predict=DECOMPOSITION_CONFIG["num_predict"],
        num_ctx=DECOMPOSITION_CONFIG["num_ctx"],
    )
    stage_failures = list(state.stage_failures)

    results = await asyncio.gather(
        *(_decomposition_stage(item, llm) for item in state.disambiguated_contents)
    )
    potential_claims: list[PotentialClaim] = []
    for claims, failure in results:
        potential_claims.extend(claims)
        if failure is not None:
            stage_failures.append(failure)

    logger.info("Extracted %s potential claims", len(potential_claims))
    return {"potential_claims": potential_claims, "stage_failures": stage_failures}
