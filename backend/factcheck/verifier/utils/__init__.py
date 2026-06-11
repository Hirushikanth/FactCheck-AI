"""Utility helpers shared by verifier nodes."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from factcheck.search import SearchHit
from factcheck.verifier.schemas import EvidenceItem
from factcheck.verifier.utils.framing import (
    extract_evaluation_frame,
    frame_tokens,
    snippet_looks_colloquial,
    snippet_matches_frame,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
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

_FRAME_MATCH_BOOST = 0.15
_COLLOQUIAL_PENALTY = 0.1


def truncate_snippet(text: str, *, max_words: int) -> str:
    """Trim a snippet to a word budget, preferring a sentence boundary."""

    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    truncated = " ".join(words[:max_words]).strip()
    sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if sentence_end == -1 and truncated.endswith((".", "!", "?")):
        sentence_end = len(truncated) - 1

    if sentence_end >= len(truncated) // 2:
        return truncated[: sentence_end + 1].strip()
    return f"{truncated}..."


def estimate_tokens(text: str) -> int:
    """Approximate token count conservatively for English snippets."""

    return int(len(text.split()) / 0.75)


def tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def token_overlap_score(claim: str, hit: SearchHit) -> float:
    claim_tokens = tokens(claim)
    hit_tokens = tokens(f"{hit.title} {hit.snippet}")
    if not claim_tokens or not hit_tokens:
        return 0.0
    return len(claim_tokens & hit_tokens) / len(claim_tokens)


def _domain(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _frame_adjusted_score(
    hit: SearchHit,
    base_score: float,
    *,
    evaluation_frame: str | None,
) -> float:
    if not evaluation_frame:
        return base_score

    frame_token_set = frame_tokens(evaluation_frame)
    snippet_text = f"{hit.title} {hit.snippet}"
    adjusted = base_score
    if snippet_matches_frame(snippet_text, frame_token_set):
        adjusted += _FRAME_MATCH_BOOST
    if snippet_looks_colloquial(snippet_text):
        adjusted -= _COLLOQUIAL_PENALTY
    return adjusted


def heuristic_prefilter_hits(
    claim: str,
    hits: list[SearchHit],
    *,
    top_n: int,
    evaluation_frame: str | None = None,
) -> list[tuple[SearchHit, float]]:
    """Drop empty/duplicate snippets and keep the best lexical candidates."""

    frame = evaluation_frame or extract_evaluation_frame(claim)
    scored_hits: list[tuple[float, int, SearchHit]] = []
    seen_snippets_by_domain: dict[str, list[set[str]]] = {}
    for original_index, hit in enumerate(hits):
        if not hit.snippet.strip():
            continue

        snippet_tokens = tokens(hit.snippet)
        domain = _domain(hit.url)
        if any(_jaccard(snippet_tokens, seen) >= 0.9 for seen in seen_snippets_by_domain.get(domain, [])):
            continue

        seen_snippets_by_domain.setdefault(domain, []).append(snippet_tokens)
        base_score = token_overlap_score(claim, hit)
        score = _frame_adjusted_score(hit, base_score, evaluation_frame=frame)
        scored_hits.append((score, original_index, hit))

    scored_hits.sort(key=lambda item: (-item[0], item[1]))
    return [(hit, score) for score, _original_index, hit in scored_hits[:top_n]]


def format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "No evidence was found."

    return "\n\n".join(
        (
            f"Source {index}: {item.url}\n"
            f"Title: {item.title or 'Untitled'}\n"
            f"Snippet: {item.snippet}"
        )
        for index, item in enumerate(evidence, start=1)
    )
