"""Input intent routing and sentence profiling for claim extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from factcheck.extractor.nodes.disambiguation import needs_contextual_disambiguation
from factcheck.extractor.utils.fidelity import sentence_has_compound_structure


ExtractionModeInput = Literal["auto", "claim", "document"]
ResolvedExtractionMode = Literal["direct_claim", "document"]
SentenceKind = Literal["checkable_fact", "opinion", "hedge", "fragment", "anaphoric"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
_FINITE_VERB_HINTS = {
    "am",
    "are",
    "be",
    "been",
    "being",
        "built",
        "can",
        "cause",
        "causes",
        "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "was",
    "were",
    "will",
    "would",
    "wrote",
}
_FRAGMENT_STARTS = {
    "about",
    "after",
    "before",
    "by",
    "during",
    "for",
    "from",
    "in",
    "of",
    "on",
    "to",
    "with",
}
_EVALUATIVE_PATTERN = re.compile(
    r"\b(should|must|ought)\b|"
    r"\b(is|are|was|were)\s+"
    r"(terrible|bad|best|worst|good|excellent|incompetent|courageous)\b|"
    r"\b(the\s+)?(best|worst)\s+(solution|option|product)\b",
    re.IGNORECASE,
)
_HEDGE_MODALS = frozenset({"could", "might", "may"})
_CONCRETE_PREDICATE_TOKENS = frozenset(
    {
        "am",
        "are",
        "cause",
        "causes",
        "did",
        "does",
        "had",
        "has",
        "have",
        "is",
        "strike",
        "strikes",
        "struck",
        "was",
        "were",
        "will",
    }
)


@dataclass(frozen=True)
class SentenceProfile:
    """Programmatic classification of a sentence for extraction routing."""

    kind: SentenceKind
    reasons: tuple[str, ...] = ()


def looks_like_complete_declarative(sentence: str) -> bool:
    """Return whether *sentence* is a complete declarative assertion in isolation."""

    stripped = sentence.strip()
    if not stripped or stripped.endswith(("?", "!")):
        return False

    tokens = [token.casefold() for token in _TOKEN_RE.findall(stripped)]
    if len(tokens) < 3 or tokens[0] in _FRAGMENT_STARTS:
        return False

    return any(
        token in _FINITE_VERB_HINTS or token.endswith(("ed", "s"))
        for token in tokens[1:]
    )


def _tokens(sentence: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(sentence)]


def _has_concrete_predicate(tokens: list[str]) -> bool:
    return any(token in _CONCRETE_PREDICATE_TOKENS for token in tokens[1:])


def _is_vague_modal_hedge(tokens: list[str]) -> bool:
    if not any(token in _HEDGE_MODALS for token in tokens):
        return False
    return not _has_concrete_predicate(tokens)


def profile_sentence(sentence: str) -> SentenceProfile:
    """Classify a sentence for extraction routing without judging truth."""

    stripped = sentence.strip()
    if not stripped:
        return SentenceProfile(kind="fragment", reasons=("empty",))

    tokens = _tokens(stripped)
    if len(tokens) < 3 or tokens[0] in _FRAGMENT_STARTS:
        return SentenceProfile(kind="fragment", reasons=("too_short_or_fragment_start",))

    if needs_contextual_disambiguation(stripped):
        return SentenceProfile(kind="anaphoric", reasons=("contextual_reference",))

    if _EVALUATIVE_PATTERN.search(stripped):
        return SentenceProfile(kind="opinion", reasons=("evaluative_language",))

    if _is_vague_modal_hedge(tokens):
        return SentenceProfile(kind="hedge", reasons=("modal_without_anchor",))

    if looks_like_complete_declarative(stripped):
        return SentenceProfile(kind="checkable_fact", reasons=("declarative_assertion",))

    return SentenceProfile(kind="fragment", reasons=("not_declarative",))


def resolve_extraction_mode(
    merged_sentences: list[str],
    *,
    forced: ExtractionModeInput = "auto",
) -> ResolvedExtractionMode:
    """Choose direct_claim vs document extraction path."""

    if forced == "claim":
        return "direct_claim"
    if forced == "document":
        return "document"

    if len(merged_sentences) != 1:
        return "document"

    profile = profile_sentence(merged_sentences[0])
    if profile.kind == "checkable_fact":
        return "direct_claim"
    return "document"


__all__ = [
    "ExtractionModeInput",
    "ResolvedExtractionMode",
    "SentenceKind",
    "SentenceProfile",
    "looks_like_complete_declarative",
    "profile_sentence",
    "resolve_extraction_mode",
    "sentence_has_compound_structure",
]
