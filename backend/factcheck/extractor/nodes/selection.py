"""Selection node for identifying verifiable sentences."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from factcheck.extractor.config import SELECTION_CONFIG
from factcheck.extractor.prompts import HUMAN_PROMPT, SELECTION_SYSTEM_PROMPT
from factcheck.extractor.schemas import ContextualSentence, ExtractorState, SelectedContent
from factcheck.extractor.utils.voting import process_with_voting
from factcheck.llm.factory import get_extractor_llm
from factcheck.llm.structured import call_llm_with_structured_output


logger = logging.getLogger(__name__)


class SelectionOutput(BaseModel):
    """Structured output for the selection stage."""

    reasoning: str = Field(
        description="Step-by-step analysis of whether the sentence contains verifiable information."
    )
    processed_sentence: str | None = Field(default=None)
    no_verifiable_claims: bool
    remains_unchanged: bool

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value


async def _single_selection_attempt(
    contextual_item: ContextualSentence,
    llm: object,
) -> tuple[bool, str | None]:
    sentence = contextual_item.original_sentence
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=SelectionOutput,
        messages=[
            ("system", SELECTION_SYSTEM_PROMPT),
            (
                "human",
                HUMAN_PROMPT.format(
                    excerpt=contextual_item.context_for_llm,
                    sentence=sentence,
                ),
            ),
        ],
        context_desc=f"selection for '{sentence}'",
    )

    if not response or not response.processed_sentence or response.no_verifiable_claims:
        return False, None

    processed = sentence if response.remains_unchanged else response.processed_sentence.strip()
    return True, processed


def _create_selected_content(
    processed_sentence: str,
    contextual_item: ContextualSentence,
) -> SelectedContent:
    return SelectedContent(
        processed_sentence=processed_sentence,
        original_context_item=contextual_item,
    )


async def selection_node(state: ExtractorState) -> dict[str, list[SelectedContent]]:
    """Keep sentences that contain at least one verifiable proposition."""

    if not state.contextual_sentences:
        return {"selected_contents": []}

    llm = get_extractor_llm(temperature=SELECTION_CONFIG["temperature"])
    selected = await process_with_voting(
        items=state.contextual_sentences,
        processor=_single_selection_attempt,
        llm=llm,
        completions=SELECTION_CONFIG["completions"],
        min_successes=SELECTION_CONFIG["min_successes"],
        result_factory=_create_selected_content,
    )
    logger.info("Selected %s/%s contextual sentences", len(selected), len(state.contextual_sentences))
    return {"selected_contents": selected}
