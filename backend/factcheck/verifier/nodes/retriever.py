"""Retriever node for collecting search evidence."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

from factcheck.search import SearchHit, search_with_fallback
from factcheck.state import ClaimResult
from factcheck.verifier.schemas import VerifierState


def _normalized_url(url: str) -> str:
    parts = urlsplit(url.strip())
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


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


async def retriever_node(
    state: VerifierState,
) -> dict[str, list[SearchHit] | ClaimResult]:
    """Retrieve and deduplicate search hits for generated queries."""

    if state.claim_result is not None:
        return {"claim_result": state.claim_result}

    if not state.search_queries:
        return {
            "raw_hits": [],
            "claim_result": _insufficient_result(state, "No search queries were generated."),
        }

    search_results = await asyncio.gather(
        *(search_with_fallback(query) for query in state.search_queries)
    )

    deduped_hits: list[SearchHit] = []
    seen_urls: set[str] = set()
    for hits, _provider_name in search_results:
        for hit in hits:
            normalized = _normalized_url(hit.url)
            if not normalized or normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            deduped_hits.append(hit)

    if not deduped_hits:
        return {
            "raw_hits": [],
            "claim_result": _insufficient_result(
                state,
                "Search returned no evidence for this claim.",
            ),
        }

    return {"raw_hits": deduped_hits}
