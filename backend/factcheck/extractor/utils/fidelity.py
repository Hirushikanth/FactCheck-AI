"""Lightweight checks for extractor claim/source fidelity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class FidelityDecision(str, Enum):
    """Programmatic fidelity decision before optional LLM audit."""

    PASS = "pass"
    FAIL = "fail"
    BORDERLINE = "borderline"


@dataclass(frozen=True)
class FidelityAssessment:
    """Result of comparing an extracted claim against its source assertion."""

    decision: FidelityDecision
    extra_terms: set[str] = field(default_factory=set)
    missing_negations: set[str] = field(default_factory=set)
    reason: str = ""


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")

_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "will",
    "with",
    "would",
}

_NEGATIONS = {"no", "not", "never", "none", "without"}


def _normalize_token(token: str) -> str:
    return token.casefold()


def _tokens(text: str) -> set[str]:
    return {_normalize_token(token) for token in _TOKEN_RE.findall(text)}


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS}


def _bracketed_context(text: str) -> str:
    return " ".join(match.group(1) for match in _BRACKET_RE.finditer(text))


def _overlap_ratio(claim_terms: set[str], source_terms: set[str]) -> float:
    if not claim_terms:
        return 1.0
    return len(claim_terms & source_terms) / len(claim_terms)


def _ordered_tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in _TOKEN_RE.findall(text)]


def _negation_scopes(source_sentence: str) -> list[set[str]]:
    """Content-token scopes immediately before each negation in the source."""

    tokens = _ordered_tokens(source_sentence)
    scopes: list[set[str]] = []
    for index, token in enumerate(tokens):
        if token not in _NEGATIONS:
            continue

        scope: list[str] = []
        for prior in reversed(tokens[max(0, index - 4) : index]):
            if prior in _STOPWORDS:
                continue
            scope.append(prior)
            if len(scope) >= 2:
                break

        if scope:
            scopes.append(set(scope))

    return scopes


def _drops_scoped_negation(claim_text: str, source_sentence: str) -> set[str]:
    """Return scoped subjects whose negation was dropped from the claim."""

    claim_terms = _content_tokens(claim_text)
    if not claim_terms:
        return set()

    claim_negations = _tokens(claim_text) & _NEGATIONS
    if claim_negations:
        return set()

    dropped: set[str] = set()
    for scope in _negation_scopes(source_sentence):
        overlap = scope & claim_terms
        if overlap:
            dropped |= overlap

    return dropped


def assess_claim_fidelity(
    *,
    claim_text: str,
    source_sentence: str,
    context_text: str | None = None,
) -> FidelityAssessment:
    """Compare an extracted claim to the source assertion without judging truth."""

    claim_terms = _content_tokens(claim_text)
    source_terms = _content_tokens(source_sentence)
    bracket_terms = _content_tokens(_bracketed_context(claim_text))
    context_terms = _content_tokens(context_text or "")

    allowed_source_terms = source_terms | bracket_terms
    extra_terms = claim_terms - allowed_source_terms

    dropped_negation_scope = _drops_scoped_negation(claim_text, source_sentence)
    if dropped_negation_scope:
        return FidelityAssessment(
            decision=FidelityDecision.FAIL,
            extra_terms=extra_terms,
            missing_negations={"not"},
            reason="Extracted claim dropped source negation for a scoped subject.",
        )

    if not extra_terms:
        return FidelityAssessment(
            decision=FidelityDecision.PASS,
            reason="Extracted claim uses only source assertion terms.",
        )

    if extra_terms <= context_terms:
        return FidelityAssessment(
            decision=FidelityDecision.BORDERLINE,
            extra_terms=extra_terms,
            reason="Extracted claim adds terms present in the original context.",
        )

    if _overlap_ratio(claim_terms, source_terms) >= 0.9 and len(extra_terms) <= 1:
        return FidelityAssessment(
            decision=FidelityDecision.BORDERLINE,
            extra_terms=extra_terms,
            reason="Extracted claim is near-verbatim with a small addition.",
        )

    return FidelityAssessment(
        decision=FidelityDecision.FAIL,
        extra_terms=extra_terms,
        reason="Extracted claim introduces terms not present in the source assertion.",
    )
