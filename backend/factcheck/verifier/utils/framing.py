"""Frame extraction and retriever ranking helpers for bracketed claims."""

from __future__ import annotations

import re

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

_COLLOQUIAL_PHRASES = (
    "commonly called",
    "everyday",
    "popular usage",
    "in the kitchen",
    "culinary",
    "colloquial",
    "generally considered",
    "often called",
)

_FRAME_SYNONYMS = {
    "botanical": {"botanical", "botanically", "botany"},
    "definition": {"definition", "definitions", "defined", "define"},
    "legal": {"legal", "legally", "statute", "statutory"},
    "medical": {"medical", "medically", "clinical", "clinically"},
}


def extract_evaluation_frame(claim_text: str) -> str | None:
    """Return bracketed evaluation frame text from a claim, if present."""

    matches = [match.group(1).strip() for match in _BRACKET_RE.finditer(claim_text)]
    if not matches:
        return None
    return " ".join(matches)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if token not in _STOPWORDS
    }


def frame_tokens(frame: str) -> set[str]:
    """Content tokens for a frame, expanded with lightweight synonyms."""

    base_tokens = _content_tokens(frame)
    expanded = set(base_tokens)
    for token in base_tokens:
        expanded.update(_FRAME_SYNONYMS.get(token, set()))
    return expanded


def snippet_matches_frame(snippet: str, frame_tokens_set: set[str]) -> bool:
    if not frame_tokens_set:
        return False
    snippet_tokens = _content_tokens(snippet)
    return bool(snippet_tokens & frame_tokens_set)


def snippet_looks_colloquial(snippet: str) -> bool:
    lowered = snippet.casefold()
    return any(phrase in lowered for phrase in _COLLOQUIAL_PHRASES)
