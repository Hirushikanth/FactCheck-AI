"""Hybrid evidence ranking node."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.search import SearchHit
from factcheck.state import ClaimResult
from factcheck.verifier.config import (
    RANKER_HEURISTIC_FALLBACK_MIN_OVERLAP,
    RANKER_HEURISTIC_TOP_N,
    RANKER_LLM_TOP_K,
    RANKER_MIN_SCORE,
    RANKER_TEMPERATURE,
)
from factcheck.verifier.prompts import (
    EVIDENCE_RANKER_HUMAN_PROMPT,
    EVIDENCE_RANKER_SYSTEM_PROMPT,
)
from factcheck.verifier.schemas import EvidenceItem, VerifierState


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


class EvidenceRanking(BaseModel):
    """LLM relevance score for a numbered candidate."""

    index: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RankerOutput(BaseModel):
    """Structured output for evidence ranking."""

    rankings: list[EvidenceRanking] = Field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def _domain(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _token_overlap_score(claim: str, hit: SearchHit) -> float:
    claim_tokens = _tokens(claim)
    hit_tokens = _tokens(f"{hit.title} {hit.snippet}")
    if not claim_tokens or not hit_tokens:
        return 0.0
    return len(claim_tokens & hit_tokens) / len(claim_tokens)


def _heuristic_prefilter_scored_hits(
    claim: str,
    hits: list[SearchHit],
    *,
    top_n: int = RANKER_HEURISTIC_TOP_N,
) -> list[tuple[SearchHit, float]]:
    """Drop weak/duplicate snippets and keep the best heuristic candidates with scores."""

    scored_hits: list[tuple[float, int, SearchHit]] = []
    seen_snippets_by_domain: dict[str, list[set[str]]] = {}
    for original_index, hit in enumerate(hits):
        if not hit.snippet.strip():
            continue

        snippet_tokens = _tokens(hit.snippet)
        domain = _domain(hit.url)
        if any(_jaccard(snippet_tokens, seen) >= 0.9 for seen in seen_snippets_by_domain.get(domain, [])):
            continue

        seen_snippets_by_domain.setdefault(domain, []).append(snippet_tokens)
        scored_hits.append((_token_overlap_score(claim, hit), original_index, hit))

    scored_hits.sort(key=lambda item: (-item[0], item[1]))
    return [(hit, score) for score, _original_index, hit in scored_hits[:top_n]]


def heuristic_prefilter_hits(
    claim: str,
    hits: list[SearchHit],
    *,
    top_n: int = RANKER_HEURISTIC_TOP_N,
) -> list[SearchHit]:
    """Drop weak/duplicate snippets and keep the best heuristic candidates."""

    scored_hits = _heuristic_prefilter_scored_hits(claim, hits, top_n=top_n)
    return [hit for hit, _score in scored_hits]


def _normalize_rankings(
    rankings: list[EvidenceRanking],
    *,
    candidate_count: int,
) -> list[EvidenceRanking]:
    if not rankings:
        return []

    indexes = [ranking.index for ranking in rankings]
    uses_one_based_indexes = 0 not in indexes and all(1 <= index <= candidate_count for index in indexes)

    normalized: list[EvidenceRanking] = []
    for ranking in rankings:
        index = ranking.index - 1 if uses_one_based_indexes else ranking.index
        if not 0 <= index < candidate_count:
            continue

        normalized.append(
            EvidenceRanking(
                index=index,
                relevance_score=max(0.0, min(1.0, ranking.relevance_score)),
                rationale=ranking.rationale,
            )
        )

    return normalized


def _heuristic_fallback_evidence(scored_candidates: list[tuple[SearchHit, float]]) -> list[EvidenceItem]:
    if not scored_candidates or scored_candidates[0][1] < RANKER_HEURISTIC_FALLBACK_MIN_OVERLAP:
        return []

    return [
        EvidenceItem(
            url=hit.url,
            title=hit.title,
            snippet=hit.snippet,
            relevance_score=score,
        )
        for hit, score in scored_candidates[:RANKER_LLM_TOP_K]
        if score >= RANKER_MIN_SCORE
    ]


def _format_candidates(candidates: list[SearchHit]) -> str:
    return "\n\n".join(
        (
            f"Candidate {index} (index={index})\n"
            f"Title: {hit.title or 'Untitled'}\n"
            f"Source: {hit.url}\n"
            f"Snippet: {hit.snippet}"
        )
        for index, hit in enumerate(candidates)
    )


def _insufficient_result(state: VerifierState, reasoning: str) -> ClaimResult:
    return {
        "claim": state.claim,
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "evidence": [],
        "sources": [],
        "reasoning": reasoning,
        "search_queries": state.search_queries,
    }


async def evidence_ranker_node(
    state: VerifierState,
) -> dict[str, list[EvidenceItem] | ClaimResult]:
    """Rank retrieved search snippets by claim relevance."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    scored_candidates = _heuristic_prefilter_scored_hits(state.claim, state.raw_hits)
    candidates = [hit for hit, _score in scored_candidates]
    if not candidates:
        return {
            "ranked_evidence": [],
            "claim_result": _insufficient_result(
                state,
                "Retrieved search results did not contain usable evidence snippets.",
            ),
        }

    llm = get_verifier_llm(temperature=RANKER_TEMPERATURE)
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=RankerOutput,
        messages=[
            ("system", EVIDENCE_RANKER_SYSTEM_PROMPT),
            (
                "human",
                EVIDENCE_RANKER_HUMAN_PROMPT.format(
                    claim=state.claim,
                    candidates=_format_candidates(candidates),
                ),
            ),
        ],
        context_desc=f"evidence ranking for '{state.claim}'",
    )

    if not response or not response.rankings:
        fallback_evidence = _heuristic_fallback_evidence(scored_candidates)
        if fallback_evidence:
            return {"ranked_evidence": fallback_evidence}

        return {
            "ranked_evidence": [],
            "claim_result": _insufficient_result(
                state,
                "The evidence ranker did not return usable scores.",
            ),
        }

    normalized_rankings = _normalize_rankings(response.rankings, candidate_count=len(candidates))
    ranked = [
        ranking
        for ranking in normalized_rankings
        if 0 <= ranking.index < len(candidates)
        and ranking.relevance_score >= RANKER_MIN_SCORE
    ]
    ranked.sort(key=lambda item: item.relevance_score, reverse=True)
    evidence_items = [
        EvidenceItem(
            url=candidates[ranking.index].url,
            title=candidates[ranking.index].title,
            snippet=candidates[ranking.index].snippet,
            relevance_score=ranking.relevance_score,
        )
        for ranking in ranked[:RANKER_LLM_TOP_K]
    ]

    if not evidence_items:
        return {
            "ranked_evidence": [],
            "claim_result": _insufficient_result(
                state,
                "Retrieved evidence was not relevant enough to verify the claim.",
            ),
        }

    return {"ranked_evidence": evidence_items}
