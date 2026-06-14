"""Sentence splitting and context-window creation for claim extraction.

Uses ``pysbd`` (Pragmatic Sentence Boundary Detector) instead of NLTK's
Punkt tokenizer.  pysbd is a rule-based segmenter that correctly handles:

* Abbreviations  — "Dr. Smith", "Inc.", "U.S.A.", "D.C."
* Decimal numbers — "version 3.14 released"
* URLs / e-mail — "visit https://example.com. Click here."
* Parenthetical fragments — often produced by web scrapers

No corpus downloads or model loading are required.
"""

from __future__ import annotations

import logging
import re

import pysbd

from factcheck.extractor.config import CONTEXT_WINDOWS
from factcheck.extractor.schemas import ContextualSentence, ExtractorState


logger = logging.getLogger(__name__)

# Module-level segmenter — pysbd.Segmenter is stateless and cheap to create,
# but there is no reason to rebuild it on every call.
_segmenter = pysbd.Segmenter(language="en", clean=False)
_KNOWN_ABBREVIATION_BOUNDARY_RE = re.compile(r"\b(D\.C\.)\s+(?=[A-Z0-9])")


def _split_known_sentence_final_abbreviations(sentence: str) -> list[str]:
    """Split cases pysbd keeps together after sentence-final abbreviations."""

    parts: list[str] = []
    start = 0
    for match in _KNOWN_ABBREVIATION_BOUNDARY_RE.finditer(sentence):
        parts.append(sentence[start : match.end(1)].strip())
        start = match.end()
    parts.append(sentence[start:].strip())
    return [part for part in parts if part]


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using pysbd.

    pysbd operates per-paragraph.  We split on blank lines first so that
    paragraph boundaries are always honoured, then apply the segmenter to each
    paragraph independently and collect the results.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # pysbd may return segments with leading/trailing whitespace.
        for segment in _segmenter.segment(paragraph):
            sentences.extend(_split_known_sentence_final_abbreviations(segment.strip()))
    return sentences


async def _sentence_splitter_and_context_creator(
    answer_text: str,
    *,
    p_sentences: int = 1,
    f_sentences: int = 1,
    include_metadata: bool = False,
    metadata: str | None = None,
) -> list[ContextualSentence]:
    """Split *answer_text* into contextual sentence windows."""

    raw_sentences = _split_sentences(answer_text)

    # Merge fragments shorter than 5 characters (e.g. stray initials that
    # slipped through) into the next sentence so the LLM always receives
    # something meaningful.
    merged_sentences: list[str] = []
    index = 0
    while index < len(raw_sentences):
        sentence = raw_sentences[index]
        while len(sentence) < 5 and index + 1 < len(raw_sentences):
            index += 1
            sentence = f"{sentence} {raw_sentences[index]}".strip()
        if sentence:
            merged_sentences.append(sentence)
        index += 1

    contextual_sentences: list[ContextualSentence] = []
    for index, sentence in enumerate(merged_sentences):
        context_parts: list[str] = []
        if include_metadata and metadata:
            context_parts.append(f"[Document Metadata: {metadata}]")

        start_index = max(0, index - p_sentences)
        if start_index < index:
            context_parts.append("[Preceding Sentences:]")
            context_parts.extend(merged_sentences[start_index:index])

        context_parts.append(f"[Sentence of Interest for current task:]\n{sentence}")

        end_index = min(len(merged_sentences), index + 1 + f_sentences)
        if index + 1 < end_index:
            context_parts.append("[Following Sentences:]")
            context_parts.extend(merged_sentences[index + 1 : end_index])

        contextual_sentences.append(
            ContextualSentence(
                original_sentence=sentence,
                context_for_llm="\n".join(context_parts),
                metadata=metadata,
                original_index=index,
            )
        )

    return contextual_sentences


async def sentence_splitter_node(state: ExtractorState) -> dict[str, list[ContextualSentence]]:
    """Split raw input into contextual sentences."""

    windows = CONTEXT_WINDOWS["selection"]
    contextual_sentences = await _sentence_splitter_and_context_creator(
        state.raw_input,
        p_sentences=windows["preceding_sentences"],
        f_sentences=windows["following_sentences"],
        include_metadata=bool(state.metadata),
        metadata=state.metadata,
    )
    return {"contextual_sentences": contextual_sentences}

