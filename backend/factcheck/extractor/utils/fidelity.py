"""Lightweight checks for extractor claim/source fidelity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache


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

_WORDNET_RESOURCES = ("wordnet", "omw-1.4")


def _normalize_token(token: str) -> str:
    return token.casefold()


def _tokens(text: str) -> set[str]:
    return {_normalize_token(token) for token in _TOKEN_RE.findall(text)}


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS}


def _bracketed_context(text: str) -> str:
    return " ".join(match.group(1) for match in _BRACKET_RE.finditer(text))


@lru_cache(maxsize=1)
def _ensure_wordnet() -> None:
    """Ensure WordNet corpora are available for morphological normalization."""

    import nltk

    for resource in _WORDNET_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    "NLTK WordNet corpora are required for fidelity checks. "
                    "Install manually with: "
                    "python -c \"import nltk; nltk.download('wordnet'); "
                    "nltk.download('omw-1.4')\""
                ) from exc


@lru_cache(maxsize=4096)
def _morphological_forms(token: str) -> frozenset[str]:
    """Return surface token plus WordNet morphological variants."""

    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    forms = {token}
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        for morph in wn._morphy(token, pos) or []:
            forms.add(morph)
    return frozenset(forms)


def _morphological_forms_union(text: str) -> set[str]:
    """Union of morphological forms for all content tokens in text."""

    forms: set[str] = set()
    for token in _content_tokens(text):
        forms |= _morphological_forms(token)
    return forms


def _token_covered(token: str, allowed_forms: set[str]) -> bool:
    return bool(_morphological_forms(token) & allowed_forms)


def _uncovered_terms(claim_terms: set[str], allowed_forms: set[str]) -> set[str]:
    return {token for token in claim_terms if not _token_covered(token, allowed_forms)}


def _overlap_ratio(claim_terms: set[str], allowed_forms: set[str]) -> float:
    if not claim_terms:
        return 1.0
    covered = sum(1 for token in claim_terms if _token_covered(token, allowed_forms))
    return covered / len(claim_terms)


def _scope_overlaps_claim(scope: set[str], claim_terms: set[str]) -> bool:
    for scope_token in scope:
        scope_forms = _morphological_forms(scope_token)
        for claim_token in claim_terms:
            if scope_forms & _morphological_forms(claim_token):
                return True
    return False


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
        if _scope_overlaps_claim(scope, claim_terms):
            dropped |= scope

    return dropped


def assess_claim_fidelity(
    *,
    claim_text: str,
    source_sentence: str,
    context_text: str | None = None,
) -> FidelityAssessment:
    """Compare an extracted claim to the source assertion without judging truth."""

    claim_terms = _content_tokens(claim_text)
    allowed_source_forms = _morphological_forms_union(source_sentence) | _morphological_forms_union(
        _bracketed_context(claim_text)
    )
    context_forms = _morphological_forms_union(context_text or "")

    extra_terms = _uncovered_terms(claim_terms, allowed_source_forms)

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

    if all(_token_covered(token, context_forms) for token in extra_terms):
        return FidelityAssessment(
            decision=FidelityDecision.BORDERLINE,
            extra_terms=extra_terms,
            reason="Extracted claim adds terms present in the original context.",
        )

    if _overlap_ratio(claim_terms, allowed_source_forms) >= 0.9 and len(extra_terms) <= 1:
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
