"""Selection node for identifying verifiable sentences."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator, model_validator

from factcheck.extractor.config import SELECTION_CONFIG
from factcheck.extractor.prompts import HUMAN_PROMPT, SELECTION_SYSTEM_PROMPT
from factcheck.extractor.schemas import ContextualSentence, ExtractorStageFailure, ExtractorState, SelectedContent
from factcheck.extractor.utils.assertion_profile import resolve_extraction_mode
from factcheck.extractor.utils.fidelity import selection_rewrite_preserves_source
from factcheck.extractor.utils.voting import process_with_voting
from factcheck.llm.extractor_structured import call_extractor_structured_output
from factcheck.llm.factory import get_extractor_llm


logger = logging.getLogger(__name__)


class SelectionOutput(BaseModel):
    """Structured output for the selection stage."""

    no_verifiable_claims: bool
    remains_unchanged: bool
    processed_sentence: str | None = Field(default=None)
    reasoning: str = Field(
        default="",
        description="Brief explanation of the selection decision.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> SelectionOutput:
        if self.no_verifiable_claims:
            return self
        if not self.processed_sentence or not self.processed_sentence.strip():
            raise ValueError("processed_sentence is required when no_verifiable_claims is false")
        return self


async def _single_selection_attempt(
    contextual_item: ContextualSentence,
    llm: object,
) -> tuple[bool, str | None]:
    sentence = contextual_item.original_sentence
    response = await call_extractor_structured_output(
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

    if not selection_rewrite_preserves_source(
        original=sentence,
        processed=response.processed_sentence.strip(),
        remains_unchanged=response.remains_unchanged,
    ):
        logger.info(
            "Selection rewrite rejected for overlap: '%s' -> '%s'",
            sentence,
            response.processed_sentence.strip(),
        )
        return False, None

    processed = sentence if response.remains_unchanged else response.processed_sentence.strip()
    return True, processed


def _create_selected_content_factory(
    preceding_by_index: dict[int, ContextualSentence],
):
    def _create_selected_content(
        processed_sentence: str,
        contextual_item: ContextualSentence,
    ) -> SelectedContent:
        return SelectedContent(
            processed_sentence=processed_sentence,
            original_context_item=contextual_item,
            preceding_context_item=preceding_by_index[contextual_item.original_index],
        )

    return _create_selected_content


def _selection_failure_sentence(item: ContextualSentence) -> str:
    return item.original_sentence


def _direct_selected_contents(state: ExtractorState) -> list[SelectedContent]:
    preceding_by_index = {item.original_index: item for item in state.preceding_context_sentences}
    factory = _create_selected_content_factory(preceding_by_index)
    return [
        factory(item.original_sentence, item)
        for item in state.contextual_sentences
    ]


async def selection_node(state: ExtractorState) -> dict[str, list[SelectedContent] | list]:
    """Keep sentences that contain at least one verifiable proposition."""

    if not state.contextual_sentences:
        return {"selected_contents": [], "resolved_extraction_mode": "document"}

    merged_sentences = [item.original_sentence for item in state.contextual_sentences]
    resolved_mode = resolve_extraction_mode(
        merged_sentences,
        forced=state.extraction_mode,
    )

    if resolved_mode == "direct_claim":
        selected = _direct_selected_contents(state)
        logger.info(
            "Selection skipped (direct_claim): %s sentence(s)",
            len(selected),
        )
        return {
            "selected_contents": selected,
            "resolved_extraction_mode": "direct_claim",
        }

    preceding_by_index = {item.original_index: item for item in state.preceding_context_sentences}
    llm = get_extractor_llm(
        temperature=SELECTION_CONFIG["temperature"],
        num_predict=SELECTION_CONFIG["num_predict"],
        num_ctx=SELECTION_CONFIG["num_ctx"],
    )
    stage_failures: list = []

    def _on_failure(item: ContextualSentence, successes: int, attempts: int) -> None:
        sentence = _selection_failure_sentence(item)
        logger.warning(
            "Selection voting dropped sentence: '%s' (%s/%s successes)",
            sentence,
            successes,
            attempts,
        )
        stage_failures.append(
            ExtractorStageFailure(
                stage="selection",
                sentence=sentence,
                reason="voting_failed",
                successes=successes,
                attempts=attempts,
            )
        )

    selected = await process_with_voting(
        items=state.contextual_sentences,
        processor=_single_selection_attempt,
        llm=llm,
        completions=SELECTION_CONFIG["completions"],
        min_successes=SELECTION_CONFIG["min_successes"],
        result_factory=_create_selected_content_factory(preceding_by_index),
        on_failure=_on_failure,
    )
    logger.info("Selected %s/%s contextual sentences", len(selected), len(state.contextual_sentences))
    return {
        "selected_contents": selected,
        "stage_failures": stage_failures,
        "resolved_extraction_mode": "document",
    }
