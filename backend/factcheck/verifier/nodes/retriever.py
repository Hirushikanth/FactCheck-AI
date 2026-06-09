"""Retriever node for collecting search evidence."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from factcheck.search import SearchHit, search_with_fallback
from factcheck.verifier.config import MAX_SNIPPET_WORDS, RANKER_HEURISTIC_TOP_N
from factcheck.verifier.schemas import EvidenceItem, VerifierState
from factcheck.verifier.utils import (
    estimate_tokens,
    heuristic_prefilter_hits,
    truncate_snippet,
)


def _truncate_snippet(text: str, max_words: int = MAX_SNIPPET_WORDS) -> str:
    return truncate_snippet(text, max_words=max_words)


def _estimate_tokens(text: str) -> int:
    return estimate_tokens(text)


def _normalized_url(url: str) -> str:
    parts = urlsplit(url.strip())
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


async def retriever_node(
    state: VerifierState,
) -> dict[str, list[EvidenceItem] | int]:
    """Retrieve and deduplicate search hits for the current query."""

    if state.claim_result is not None:
        return {"evidence": []}

    if not state.current_query or state.estimated_evidence_tokens >= state.max_evidence_tokens:
        return {"evidence": [], "estimated_evidence_tokens": state.estimated_evidence_tokens}

    hits, _provider_name = await search_with_fallback(state.current_query)

    existing_urls = {_normalized_url(item.url) for item in state.evidence}
    deduped_hits: list[SearchHit] = []
    seen_urls = set(existing_urls)
    for hit in hits:
        normalized = _normalized_url(hit.url)
        if not normalized or normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        deduped_hits.append(hit)

    if not deduped_hits:
        return {"evidence": [], "estimated_evidence_tokens": state.estimated_evidence_tokens}

    new_evidence: list[EvidenceItem] = []
    new_tokens = 0
    for hit, score in heuristic_prefilter_hits(
        state.claim_text,
        deduped_hits,
        top_n=RANKER_HEURISTIC_TOP_N,
    ):
        snippet = _truncate_snippet(hit.snippet)
        token_count = _estimate_tokens(snippet)
        if state.estimated_evidence_tokens + new_tokens + token_count > state.max_evidence_tokens:
            break

        new_tokens += token_count
        new_evidence.append(
            EvidenceItem(
                url=hit.url,
                title=hit.title,
                snippet=snippet,
                relevance_score=score,
            )
        )

    return {
        "evidence": new_evidence,
        "estimated_evidence_tokens": state.estimated_evidence_tokens + new_tokens,
    }
