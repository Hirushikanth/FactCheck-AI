"""Frame extraction and verdict guardrails for technically framed claims."""

from __future__ import annotations

import re

from factcheck.verifier.schemas import EvidenceItem


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_NEGATION_RE = re.compile(r"\bnot\b", re.IGNORECASE)

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

_NEGATIONS = {"no", "not", "never", "none", "without"}

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


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


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


def _claim_body_without_brackets(claim_text: str) -> str:
    return _BRACKET_RE.sub("", claim_text).strip()


def colloquial_contradicts_framed_claim(claim_text: str, snippet: str) -> bool:
    """True when a colloquial snippet affirms the positive form of a negated claim."""

    claim_body = _claim_body_without_brackets(claim_text)
    if "not" not in claim_body.casefold():
        return False

    frame = extract_evaluation_frame(claim_text)
    if frame and snippet_matches_frame(snippet, frame_tokens(frame)):
        return False

    positive_claim = _NEGATION_RE.sub("", claim_body, count=1)
    positive_claim = re.sub(r"\s+", " ", positive_claim).strip(" ,.")
    positive_tokens = _content_tokens(positive_claim)
    snippet_tokens = _content_tokens(snippet)
    if not positive_tokens:
        return False

    overlap_ratio = len(positive_tokens & snippet_tokens) / len(positive_tokens)
    snippet_has_negation = bool(_tokens(snippet) & _NEGATIONS)
    return overlap_ratio >= 0.6 and not snippet_has_negation


def frame_aligned_refutes_framed_claim(
    claim_text: str,
    snippet: str,
    frame_tokens_set: set[str],
) -> bool:
    """True when frame-aligned evidence contradicts the framed claim."""

    if not snippet_matches_frame(snippet, frame_tokens_set):
        return False

    claim_body = _claim_body_without_brackets(claim_text)
    if "not" not in claim_body.casefold():
        return False

    snippet_has_negation = bool(_tokens(snippet) & _NEGATIONS)
    if snippet_has_negation:
        return False

    return colloquial_contradicts_framed_claim(claim_text, snippet) or _content_tokens(
        claim_body
    ) & _content_tokens(snippet)


def adjust_verdict_for_framing(
    *,
    claim_text: str,
    verdict: str,
    confidence: float,
    reasoning: str,
    evidence: list[EvidenceItem],
) -> tuple[str, float, str]:
    """Downgrade colloquial-only REFUTED verdicts on framed claims."""

    frame = extract_evaluation_frame(claim_text)
    if verdict != "REFUTED" or not frame:
        return verdict, confidence, reasoning

    frame_token_set = frame_tokens(frame)
    has_frame_evidence = any(
        snippet_matches_frame(item.snippet, frame_token_set) for item in evidence
    )
    has_frame_refutation = any(
        frame_aligned_refutes_framed_claim(claim_text, item.snippet, frame_token_set)
        for item in evidence
    )
    has_colloquial_conflict = any(
        snippet_looks_colloquial(item.snippet)
        and colloquial_contradicts_framed_claim(claim_text, item.snippet)
        for item in evidence
    )

    if has_frame_refutation:
        return verdict, confidence, reasoning

    if has_colloquial_conflict:
        adjusted_reasoning = (
            f"{reasoning} Colloquial evidence conflicts with technical framing; verdict adjusted."
        )
        return "CONFLICTING_EVIDENCE", min(confidence, 0.7), adjusted_reasoning

    if not has_frame_evidence:
        adjusted_reasoning = (
            f"{reasoning} No frame-aligned evidence found; verdict adjusted."
        )
        return "INSUFFICIENT_EVIDENCE", min(confidence, 0.5), adjusted_reasoning

    return verdict, confidence, reasoning
