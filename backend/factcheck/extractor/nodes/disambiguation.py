"""Disambiguation node for decontextualizing selected content."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from factcheck.extractor.config import DISAMBIGUATION_CONFIG
from factcheck.extractor.prompts import DISAMBIGUATION_SYSTEM_PROMPT, HUMAN_PROMPT
from factcheck.extractor.schemas import DisambiguatedContent, ExtractorState, SelectedContent
from factcheck.extractor.utils.text import remove_following_sentences
from factcheck.extractor.utils.voting import process_with_voting
from factcheck.llm.factory import get_extractor_llm
from factcheck.llm.structured import call_llm_with_structured_output


logger = logging.getLogger(__name__)


class DisambiguationOutput(BaseModel):
    """Structured output for the disambiguation stage."""

    disambiguated_sentence: str | None = Field(default=None)
    cannot_be_disambiguated: bool


_CONTEXTUAL_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"he|she|it|they|them|his|her|hers|its|their|theirs|"
    r"this|that|these|those|here|there|then|"
    r"today|yesterday|tomorrow|"
    r"last\s+(?:year|month|week|winter|summer|spring|fall|autumn)|"
    r"next\s+(?:year|month|week|winter|summer|spring|fall|autumn)|"
    r"at\s+the\s+time|the\s+company|the\s+organization"
    r")\b",
    re.IGNORECASE,
)


def _needs_contextual_disambiguation(sentence: str) -> bool:
    """Return whether a sentence has obvious references needing context."""

    return bool(_CONTEXTUAL_REFERENCE_PATTERN.search(sentence))


async def _single_disambiguation_attempt(
    selected_item: SelectedContent,
    llm: object,
) -> tuple[bool, str | None]:
    sentence = selected_item.processed_sentence
    if not _needs_contextual_disambiguation(sentence):
        return True, sentence.strip()

    context = remove_following_sentences(selected_item.original_context_item.context_for_llm)
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
    disambiguated = await process_with_voting(
        items=state.selected_contents,
        processor=_single_disambiguation_attempt,
        llm=llm,
        completions=DISAMBIGUATION_CONFIG["completions"],
        min_successes=DISAMBIGUATION_CONFIG["min_successes"],
        result_factory=_create_disambiguated_content,
    )
    logger.info("Disambiguated %s/%s items", len(disambiguated), len(state.selected_contents))
    return {"disambiguated_contents": disambiguated}
