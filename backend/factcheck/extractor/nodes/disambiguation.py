"""Disambiguation node for decontextualizing selected content."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field, field_validator

from factcheck.extractor.config import DISAMBIGUATION_CONFIG
from factcheck.extractor.prompts import (
    BATCH_DISAMBIGUATION_HUMAN_PROMPT,
    DISAMBIGUATION_SYSTEM_PROMPT,
    HUMAN_PROMPT,
)
from factcheck.extractor.schemas import DisambiguatedContent, ExtractorState, SelectedContent
from factcheck.extractor.utils.voting import process_batch_with_voting, process_with_voting
from factcheck.llm.factory import get_extractor_llm
from factcheck.llm.structured import call_llm_with_structured_output


logger = logging.getLogger(__name__)


class DisambiguationOutput(BaseModel):
    """Structured output for the disambiguation stage."""

    reasoning: str = Field(
        description="Step-by-step analysis of contextual references and linguistic ambiguity."
    )
    disambiguated_sentence: str | None = Field(default=None)
    cannot_be_disambiguated: bool

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value


class BatchDisambiguationItemOutput(BaseModel):
    """Structured output for one item in a batch disambiguation call."""

    original_index: int
    reasoning: str = Field(
        description="Step-by-step analysis of contextual references and linguistic ambiguity."
    )
    disambiguated_sentence: str | None = Field(default=None)
    cannot_be_disambiguated: bool

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> object:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None and str(item).strip())
        return value


class BatchDisambiguationOutput(BaseModel):
    """Structured output for a batched disambiguation attempt."""

    results: list[BatchDisambiguationItemOutput] = Field(default_factory=list)


_CONTEXTUAL_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"he|she|it|they|them|his|her|hers|its|their|theirs|"
    r"this|that|these|those|then|"
    r"today|yesterday|tomorrow|"
    r"last\s+(?:year|month|week|winter|summer|spring|fall|autumn)|"
    r"next\s+(?:year|month|week|winter|summer|spring|fall|autumn)|"
    r"at\s+the\s+time|the\s+company|the\s+organization"
    r")\b",
    re.IGNORECASE,
)

_LOCATIVE_THERE_PATTERN = re.compile(
    r"\bthere\b(?!\s+(?:is|are|was|were|will|would|has|have|had|"
    r"might|could|should|must|may|can|shall|be|been|being))",
    re.IGNORECASE,
)

_LOCATIVE_HERE_PATTERN = re.compile(
    r"\bhere\b(?!\s*[,])",
    re.IGNORECASE,
)
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
_MULTI_WORD_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _format_batch_disambiguation_items(items: list[SelectedContent]) -> str:
    blocks: list[str] = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"Sentence #{item.original_context_item.original_index}",
                    "Excerpt:",
                    item.preceding_context_item.context_for_llm,
                    "",
                    "Sentence:",
                    item.processed_sentence,
                ]
            )
        )
    return "\n\n".join(blocks)


def _needs_contextual_disambiguation(sentence: str) -> bool:
    """Return whether a sentence has obvious references needing context.

    Checks pronouns, demonstratives, temporal references, and locative uses of
    ``there``/``here`` while skipping existential ``there is/are`` and discourse
    ``here,`` openers common in factual text.
    """
    if _CONTEXTUAL_REFERENCE_PATTERN.search(sentence):
        return True
    if _LOCATIVE_THERE_PATTERN.search(sentence):
        return True
    if _LOCATIVE_HERE_PATTERN.search(sentence):
        return True
    return False


def _needs_entity_expansion(selected_item: SelectedContent) -> bool:
    """Return whether context may expand a partial name or acronym."""

    sentence = selected_item.processed_sentence
    context = selected_item.preceding_context_item.context_for_llm
    multi_word_names = _MULTI_WORD_NAME_PATTERN.findall(context)
    for acronym in _ACRONYM_PATTERN.findall(sentence):
        for full_name in multi_word_names:
            initials = "".join(token[0].upper() for token in full_name.split())
            if initials == acronym and full_name not in sentence:
                return True
        if re.search(
            rf"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\s*\({re.escape(acronym)}\)",
            context,
        ) or re.search(
            rf"\b{re.escape(acronym)}\s*\([A-Za-z][^)]+\)",
            context,
        ):
            return True

    sentence_tokens = {
        token.casefold().removesuffix("'s").removesuffix("s'")
        for token in _WORD_PATTERN.findall(sentence)
    }
    if not sentence_tokens:
        return False

    for full_name in multi_word_names:
        name_tokens = full_name.split()
        if len(name_tokens) < 2 or full_name in sentence:
            continue
        normalized_name_tokens = {
            token.casefold().removesuffix("'s").removesuffix("s'") for token in name_tokens
        }
        if normalized_name_tokens & sentence_tokens:
            return True
    return False


async def _single_disambiguation_attempt(
    selected_item: SelectedContent,
    llm: object,
) -> tuple[bool, str | None]:
    sentence = selected_item.processed_sentence
    if not _needs_contextual_disambiguation(sentence):
        return True, sentence.strip()

    context = selected_item.preceding_context_item.context_for_llm
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=DisambiguationOutput,
        messages=[
            ("system", DISAMBIGUATION_SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT.format(excerpt=context, sentence=sentence)),
        ],
        context_desc=f"disambiguation for '{sentence}'",
    )

    if not response or not response.disambiguated_sentence or response.cannot_be_disambiguated:
        return False, None

    disambiguated = response.disambiguated_sentence.strip()
    return True, disambiguated


async def _batch_disambiguation_attempt(
    selected_items: list[SelectedContent],
    llm: object,
) -> dict[int, tuple[bool, str | None]]:
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=BatchDisambiguationOutput,
        messages=[
            ("system", DISAMBIGUATION_SYSTEM_PROMPT),
            (
                "human",
                BATCH_DISAMBIGUATION_HUMAN_PROMPT.format(
                    items=_format_batch_disambiguation_items(selected_items),
                ),
            ),
        ],
        context_desc=f"disambiguation batch for {len(selected_items)} sentences",
    )

    if not response:
        return {}

    results: dict[int, tuple[bool, str | None]] = {}
    available_indexes = {item.original_context_item.original_index for item in selected_items}
    for item_output in response.results:
        if item_output.original_index not in available_indexes:
            continue
        if not item_output.disambiguated_sentence or item_output.cannot_be_disambiguated:
            results[item_output.original_index] = (False, None)
            continue
        results[item_output.original_index] = (True, item_output.disambiguated_sentence.strip())
    return results


def _create_disambiguated_content(
    disambiguated_sentence: str,
    selected_item: SelectedContent,
) -> DisambiguatedContent:
    return DisambiguatedContent(
        disambiguated_sentence=disambiguated_sentence,
        original_selected_item=selected_item,
    )


async def disambiguation_node(state: ExtractorState) -> dict[str, list[DisambiguatedContent]]:
    """Resolve references and discard content with unresolved ambiguity."""

    if not state.selected_contents:
        return {"disambiguated_contents": []}

    llm = get_extractor_llm(temperature=DISAMBIGUATION_CONFIG["temperature"])
    passthrough: dict[int, DisambiguatedContent] = {}
    ambiguous_items: list[SelectedContent] = []
    for item in state.selected_contents:
        if _needs_contextual_disambiguation(item.processed_sentence) or _needs_entity_expansion(item):
            ambiguous_items.append(item)
            continue
        passthrough[item.original_context_item.original_index] = _create_disambiguated_content(
            item.processed_sentence.strip(),
            item,
        )

    batched = []
    if ambiguous_items:
        batched = await process_batch_with_voting(
            items=ambiguous_items,
            batch_processor=_batch_disambiguation_attempt,
            llm=llm,
            completions=DISAMBIGUATION_CONFIG["completions"],
            min_successes=DISAMBIGUATION_CONFIG["min_successes"],
            result_factory=_create_disambiguated_content,
            item_key=lambda item: item.original_context_item.original_index,
        )

    disambiguated_by_index = {
        item.original_selected_item.original_context_item.original_index: item
        for item in batched
    }
    unresolved_items = [
        item
        for item in ambiguous_items
        if item.original_context_item.original_index not in disambiguated_by_index
    ]
    if unresolved_items:
        fallback_disambiguated = await process_with_voting(
            items=unresolved_items,
            processor=_single_disambiguation_attempt,
            llm=llm,
            completions=DISAMBIGUATION_CONFIG["completions"],
            min_successes=DISAMBIGUATION_CONFIG["min_successes"],
            result_factory=_create_disambiguated_content,
        )
        for content in fallback_disambiguated:
            disambiguated_by_index[
                content.original_selected_item.original_context_item.original_index
            ] = content

    disambiguated_by_index.update(passthrough)
    disambiguated = [
        disambiguated_by_index[item.original_context_item.original_index]
        for item in state.selected_contents
        if item.original_context_item.original_index in disambiguated_by_index
    ]
    logger.info("Disambiguated %s/%s items", len(disambiguated), len(state.selected_contents))
    return {"disambiguated_contents": disambiguated}
