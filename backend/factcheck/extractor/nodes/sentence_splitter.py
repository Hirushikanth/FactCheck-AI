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
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

_MIN_MEANINGFUL_WORDS = 3
_STANDALONE_RESPONSES = frozenset({"no", "yes", "ok", "true", "false"})
_COPULAS_AND_AUXILIARIES = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
    }
)
_IRREGULAR_VERB_FORMS = frozenset(
    {
        "began",
        "built",
        "came",
        "did",
        "found",
        "gave",
        "grew",
        "had",
        "kept",
        "left",
        "made",
        "put",
        "ran",
        "read",
        "said",
        "saw",
        "sent",
        "took",
        "told",
        "went",
        "won",
        "wrote",
    }
)


def _is_meaningful_fragment(sentence: str) -> bool:
    """Return whether a segment is likely a standalone assertion worth keeping."""
    words = _WORD_RE.findall(sentence)
    if not words:
        return False

    if len(words) == 1 and words[0].lower().rstrip(".") in _STANDALONE_RESPONSES:
        return True

    if len(words) < _MIN_MEANINGFUL_WORDS:
        return False

    words_lower = [word.lower() for word in words]
    if any(word in _COPULAS_AND_AUXILIARIES for word in words_lower):
        return True

    if any(word in _IRREGULAR_VERB_FORMS for word in words_lower):
        return True

    for word in words_lower[1:]:
        if word.endswith(("ed", "ing")) and len(word) > 4:
            return True
        if word.endswith("s") and len(word) > 3 and word not in _COPULAS_AND_AUXILIARIES:
            return True

    return False


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

    # Merge non-assertion fragments (abbreviation debris, stray tokens) into the
    # next sentence using predicate detection rather than character count.
    merged_sentences: list[str] = []
    index = 0
    while index < len(raw_sentences):
        sentence = raw_sentences[index]
        while not _is_meaningful_fragment(sentence) and index + 1 < len(raw_sentences):
            index += 1
            sentence = f"{sentence} {raw_sentences[index]}".strip()
        if sentence.strip():
            merged_sentences.append(sentence.strip())
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
    """Split raw input into per-stage contextual sentences.

    Selection receives bidirectional windows. Disambiguation and decomposition
    both consume ``preceding_context_sentences``, built from the shared
    preceding-only window config.
    """
    selection_window = CONTEXT_WINDOWS["selection"]
    preceding_window = CONTEXT_WINDOWS["preceding_only"]

    contextual_sentences = await _sentence_splitter_and_context_creator(
        state.raw_input,
        p_sentences=selection_window["preceding_sentences"],
        f_sentences=selection_window["following_sentences"],
        include_metadata=bool(state.metadata),
        metadata=state.metadata,
    )
    preceding_context_sentences = await _sentence_splitter_and_context_creator(
        state.raw_input,
        p_sentences=preceding_window["preceding_sentences"],
        f_sentences=preceding_window["following_sentences"],
        include_metadata=bool(state.metadata),
        metadata=state.metadata,
    )
    return {
        "contextual_sentences": contextual_sentences,
        "preceding_context_sentences": preceding_context_sentences,
    }

