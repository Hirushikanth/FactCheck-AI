"""Selection node for identifying verifiable sentences."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from factcheck.extractor.config import SELECTION_CONFIG
from factcheck.extractor.prompts import (
    BATCH_SELECTION_HUMAN_PROMPT,
    HUMAN_PROMPT,
    SELECTION_SYSTEM_PROMPT,
)
from factcheck.extractor.schemas import ContextualSentence, ExtractorState, SelectedContent
from factcheck.extractor.utils.voting import process_batch_with_voting, process_with_voting
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


class BatchSelectionItemOutput(BaseModel):
    """Structured output for one item in a batch selection call."""

    original_index: int
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


class BatchSelectionOutput(BaseModel):
    """Structured output for a batched selection attempt."""

    results: list[BatchSelectionItemOutput] = Field(default_factory=list)


def _format_batch_selection_items(items: list[ContextualSentence]) -> str:
    blocks: list[str] = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"Sentence #{item.original_index}",
                    "Excerpt:",
                    item.context_for_llm,
                    "",
                    "Sentence:",
                    item.original_sentence,
                ]
            )
        )
    return "\n\n".join(blocks)


async def _batch_selection_attempt(
    contextual_items: list[ContextualSentence],
    llm: object,
) -> dict[int, tuple[bool, str | None]]:
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=BatchSelectionOutput,
        messages=[
            ("system", SELECTION_SYSTEM_PROMPT),
            (
                "human",
                BATCH_SELECTION_HUMAN_PROMPT.format(
                    items=_format_batch_selection_items(contextual_items),
                ),
            ),
        ],
        context_desc=f"selection batch for {len(contextual_items)} sentences",
    )

    if not response:
        return {}

    results: dict[int, tuple[bool, str | None]] = {}
    original_sentence_by_index = {
        item.original_index: item.original_sentence for item in contextual_items
    }
    for item_output in response.results:
        original_sentence = original_sentence_by_index.get(item_output.original_index)
        if original_sentence is None:
            continue
        if not item_output.processed_sentence or item_output.no_verifiable_claims:
            results[item_output.original_index] = (False, None)
            continue
        processed = (
            original_sentence
            if item_output.remains_unchanged
            else item_output.processed_sentence.strip()
        )
        results[item_output.original_index] = (True, processed)
    return results


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


async def selection_node(state: ExtractorState) -> dict[str, list[SelectedContent]]:
    """Keep sentences that contain at least one verifiable proposition."""

    if not state.contextual_sentences:
        return {"selected_contents": []}

    preceding_by_index = {item.original_index: item for item in state.preceding_context_sentences}
    llm = get_extractor_llm(temperature=SELECTION_CONFIG["temperature"])
    selected = await process_batch_with_voting(
        items=state.contextual_sentences,
        batch_processor=_batch_selection_attempt,
        llm=llm,
        completions=SELECTION_CONFIG["completions"],
        min_successes=SELECTION_CONFIG["min_successes"],
        result_factory=_create_selected_content_factory(preceding_by_index),
        item_key=lambda item: item.original_index,
    )
    selected_by_index = {
        content.original_context_item.original_index: content for content in selected
    }
    unresolved_items = [
        item for item in state.contextual_sentences if item.original_index not in selected_by_index
    ]
    if unresolved_items:
        fallback_selected = await process_with_voting(
            items=unresolved_items,
            processor=_single_selection_attempt,
            llm=llm,
            completions=SELECTION_CONFIG["completions"],
            min_successes=SELECTION_CONFIG["min_successes"],
            result_factory=_create_selected_content_factory(preceding_by_index),
        )
        for content in fallback_selected:
            selected_by_index[content.original_context_item.original_index] = content

    ordered_selected = [
        selected_by_index[item.original_index]
        for item in state.contextual_sentences
        if item.original_index in selected_by_index
    ]
    logger.info("Selected %s/%s contextual sentences", len(ordered_selected), len(state.contextual_sentences))
    return {"selected_contents": ordered_selected}
