"""Heuristic signals for post-LLM verdict guardrails."""

from __future__ import annotations

import re

from factcheck.verifier.schemas import EvidenceItem


_CONTRADICTION_PHRASES = (
    "do not overload",
    "does not overload",
    "don't overload",
    "doesn't overload",
    "cannot overload",
    "can't overload",
    "immune system can handle",
    "immune system is capable",
    "safe to receive multiple",
    "safe to get multiple",
    "no evidence of overload",
    "antigenic overload does not exist",
    "antigenic overload does not",
    "not overwhelm the immune",
    "does not overwhelm",
    "don't overwhelm",
    "doesn't overwhelm",
    "myth that vaccines",
    "debunked",
    "no scientific basis",
)

_CLAIM_OVERLOAD_PATTERNS = (
    re.compile(r"\boverload\b", re.IGNORECASE),
    re.compile(r"\boverwhelm\b", re.IGNORECASE),
    re.compile(r"\btoo many\b.*\bvaccin", re.IGNORECASE),
)

_THRESHOLD_ASPECT_PATTERNS = (
    re.compile(r"\bexact\b", re.IGNORECASE),
    re.compile(r"\bthreshold\b", re.IGNORECASE),
    re.compile(r"\bspecific number\b", re.IGNORECASE),
    re.compile(r"\bmore than\b.*\b(two|2|three|3|\d+)\b", re.IGNORECASE),
    re.compile(r"\bper year\b", re.IGNORECASE),
    re.compile(r"\bannual\b", re.IGNORECASE),
    re.compile(r"\bquantif", re.IGNORECASE),
)

_REASONING_CONTRADICTION_PHRASES = (
    "contradict",
    "conflicting",
    "refute",
    "does not support",
    "do not support",
    "doesn't support",
    "don't support",
    "not supported",
)

_AUTHORITATIVE_TIERS = frozenset({"high", "medium"})


def claim_asserts_overload_or_harm(claim_text: str) -> bool:
    """Return True when the claim asserts immune overload or similar harm."""
    return any(pattern.search(claim_text) for pattern in _CLAIM_OVERLOAD_PATTERNS)


def snippet_contradicts_overload(snippet: str) -> bool:
    """Return True when a snippet contains phrases refuting immune overload."""
    lowered = snippet.casefold()
    return any(phrase in lowered for phrase in _CONTRADICTION_PHRASES)


def missing_aspects_only_reference_thresholds(missing_aspects: list[str]) -> bool:
    """Return True when all missing aspects refer to numeric thresholds or exact figures."""
    if not missing_aspects:
        return False
    return all(
        any(pattern.search(aspect) for pattern in _THRESHOLD_ASPECT_PATTERNS)
        for aspect in missing_aspects
    )


def reasoning_suggests_contradiction(reasoning: str) -> bool:
    """Return True when LLM reasoning language implies contradictory evidence."""
    lowered = reasoning.casefold()
    return any(phrase in lowered for phrase in _REASONING_CONTRADICTION_PHRASES)


def _relevant_evidence_items(
    evidence: list[EvidenceItem],
    influential_indices: list[int],
) -> list[tuple[int, EvidenceItem]]:
    """Select influential items first, otherwise top authoritative items by relevance."""
    if influential_indices:
        selected: list[tuple[int, EvidenceItem]] = []
        for index in influential_indices:
            if 1 <= index <= len(evidence):
                selected.append((index, evidence[index - 1]))
        return selected

    ranked = sorted(
        enumerate(evidence, start=1),
        key=lambda item: (-item[1].relevance_score, item[0]),
    )
    return [(index, item) for index, item in ranked if item.credibility_tier in _AUTHORITATIVE_TIERS]


def count_authoritative_contradictions(
    *,
    claim_text: str,
    evidence: list[EvidenceItem],
    influential_indices: list[int],
) -> list[int]:
    """Return 1-based indices of authoritative excerpts that contradict overload claims."""
    if not claim_asserts_overload_or_harm(claim_text):
        return []

    contradicting: list[int] = []
    for index, item in _relevant_evidence_items(evidence, influential_indices):
        if item.credibility_tier not in _AUTHORITATIVE_TIERS:
            continue
        if snippet_contradicts_overload(item.snippet):
            contradicting.append(index)
    return contradicting
