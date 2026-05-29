"""Sentence splitting and context-window creation for claim extraction."""

from __future__ import annotations

import logging

import nltk

from factcheck.extractor.config import CONTEXT_WINDOWS
from factcheck.extractor.schemas import ContextualSentence, ExtractorState


logger = logging.getLogger(__name__)


def ensure_nltk_resources() -> None:
    """Ensure the Punkt sentence tokenizer is available."""

    resource_names = {"tokenizers/punkt_tab": "punkt_tab", "tokenizers/punkt": "punkt"}
    missing_resources: list[str] = []
    for resource, download_name in resource_names.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            missing_resources.append(download_name)

    for resource in missing_resources:
        logger.info("Downloading NLTK resource %s for sentence splitting", resource)
        nltk.download(resource, quiet=True)


async def _sentence_splitter_and_context_creator(
    answer_text: str,
    *,
    p_sentences: int = 1,
    f_sentences: int = 1,
    include_metadata: bool = False,
    metadata: str | None = None,
) -> list[ContextualSentence]:
    """Split text into contextual sentence windows."""

    ensure_nltk_resources()
    paragraphs = [paragraph.strip() for paragraph in answer_text.split("\n") if paragraph.strip()]

    raw_sentences: list[str] = []
    for paragraph in paragraphs:
        raw_sentences.extend(nltk.sent_tokenize(paragraph))

    merged_sentences: list[str] = []
    index = 0
    while index < len(raw_sentences):
        sentence = raw_sentences[index].strip()
        while len(sentence) < 5 and index + 1 < len(raw_sentences):
            index += 1
            sentence = f"{sentence} {raw_sentences[index].strip()}".strip()
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
