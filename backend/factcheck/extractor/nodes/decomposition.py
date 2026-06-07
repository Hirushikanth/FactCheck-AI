"""Decomposition node for atomic claim extraction."""

from __future__ import annotations

import asyncio
import itertools
import logging

from pydantic import BaseModel, Field

from factcheck.extractor.config import DECOMPOSITION_CONFIG
from factcheck.extractor.nodes.reasoning import ReasoningText
from factcheck.extractor.prompts import DECOMPOSITION_SYSTEM_PROMPT, HUMAN_PROMPT
from factcheck.extractor.schemas import DisambiguatedContent, ExtractorState, PotentialClaim
from factcheck.extractor.utils.text import remove_following_sentences
from factcheck.llm.factory import get_extractor_llm
from factcheck.llm.structured import call_llm_with_structured_output


logger = logging.getLogger(__name__)


class DecompositionOutput(BaseModel):
    """Structured output for the decomposition stage."""

    reasoning: ReasoningText
    claims: list[str] = Field(default_factory=list)
    no_claims: bool


async def _decomposition_stage(item: DisambiguatedContent) -> list[PotentialClaim]:
    sentence = item.disambiguated_sentence
    original_context = item.original_selected_item.original_context_item.context_for_llm
    context = remove_following_sentences(original_context)
    llm = get_extractor_llm(temperature=DECOMPOSITION_CONFIG["temperature"])

    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=DecompositionOutput,
        messages=[
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT.format(excerpt=context, sentence=sentence)),
        ],
        context_desc=f"decomposition for '{sentence}'",
    )

    if not response or response.no_claims or not response.claims:
        return []

    original = item.original_selected_item.original_context_item
    claims = [claim.strip() for claim in response.claims if claim.strip()]
    return [
        PotentialClaim(
            claim_text=claim,
            disambiguated_sentence=sentence,
            original_sentence=original.original_sentence,
            original_index=original.original_index,
        )
        for claim in claims
    ]


async def decomposition_node(state: ExtractorState) -> dict[str, list[PotentialClaim]]:
    """Break disambiguated sentences into independently verifiable claims."""

    if not state.disambiguated_contents:
        return {"potential_claims": []}

    # asyncio.gather schedules all coroutines concurrently, but the semaphore
    # inside call_llm_with_structured_output (factcheck.llm.concurrency) means
    # at most OLLAMA_CONCURRENCY requests hit the server at once.  The gather
    # itself is safe — idle coroutines just wait on the semaphore.
    claims = await asyncio.gather(
        *(_decomposition_stage(item) for item in state.disambiguated_contents)
    )
    potential_claims = list(itertools.chain.from_iterable(claims))
    logger.info("Extracted %s potential claims", len(potential_claims))
    return {"potential_claims": potential_claims}
