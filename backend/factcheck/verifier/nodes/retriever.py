"""Retriever node for collecting search evidence."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlsplit, urlunsplit

from factcheck.config import AppSettings, get_settings
from factcheck.http.pinned_fetch import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    fetch_html_pinned,
)
from factcheck.http.url_policy import UrlPolicyError, is_safe_citation_url
from factcheck.search import SearchHit, search_with_fallback
from factcheck.verifier.config import FULL_PAGE_FETCH_TOP_N, MAX_SNIPPET_WORDS, RANKER_HEURISTIC_TOP_N
from factcheck.verifier.schemas import EvidenceItem, VerifierState
from factcheck.verifier.utils import (
    estimate_formatted_evidence_tokens,
    heuristic_prefilter_hits,
    truncate_snippet,
)
from factcheck.verifier.utils.credibility import classify_domain
from factcheck.verifier.utils.framing import extract_evaluation_frame


logger = logging.getLogger(__name__)

_MAX_FETCHED_CHARS = 3000
_SKIP_FETCH_DOMAINS = frozenset({
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "linkedin.com",
    "web.archive.org",
})


def _truncate_snippet(text: str, max_words: int = MAX_SNIPPET_WORDS) -> str:
    return truncate_snippet(text, max_words=max_words)


def _normalized_url(url: str) -> str:
    parts = urlsplit(url.strip())
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def _domain_from_url(url: str) -> str:
    return urlsplit(url).netloc.lower().lstrip("www.")


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML without any heavy library."""
    html = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(r"<[^>]+>", " ", html)
    html = (
        html.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", html).strip()


async def _resolve_page_text(hit: SearchHit, settings: AppSettings) -> str | None:
    """Resolve full-page text from provider content or a pinned self-fetch."""

    if hit.page_text:
        return hit.page_text[:_MAX_FETCHED_CHARS]

    if settings.full_page_fetch_mode == "off":
        return None

    domain = _domain_from_url(hit.url)
    if domain in _SKIP_FETCH_DOMAINS:
        logger.debug("[retriever] Skipping fetch for blocked domain: %s", domain)
        return None

    try:
        html = await fetch_html_pinned(hit.url, timeout=DEFAULT_FETCH_TIMEOUT_SECONDS)
    except UrlPolicyError as exc:
        logger.debug("[retriever] Blocked unsafe fetch URL %s: %s", hit.url, exc)
        return None
    except Exception as exc:
        logger.debug("[retriever] Failed to fetch %s: %s", hit.url, exc)
        return None

    text = _extract_text_from_html(html)
    return text[:_MAX_FETCHED_CHARS] if text else None


def _active_queries(state: VerifierState) -> list[str]:
    if state.current_queries:
        return state.current_queries
    if state.current_query:
        return [state.current_query]
    return []


async def _search_all_queries(queries: list[str]) -> list[SearchHit]:
    """Run searches for all queries in parallel and merge hits."""
    if not queries:
        return []

    results = await asyncio.gather(
        *[search_with_fallback(query) for query in queries],
        return_exceptions=True,
    )

    merged: list[SearchHit] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("[retriever] Search failed: %s", result)
            continue
        hits, _provider_name = result
        merged.extend(hits)

    return merged


async def retriever_node(
    state: VerifierState,
) -> dict[str, list[EvidenceItem] | int]:
    """Retrieve and deduplicate search hits, with full-page fetch for top results."""

    if state.claim_result is not None:
        return {"evidence": []}

    queries = _active_queries(state)
    if not queries or state.estimated_evidence_tokens >= state.max_evidence_tokens:
        return {"evidence": [], "estimated_evidence_tokens": state.estimated_evidence_tokens}

    hits = await _search_all_queries(queries)

    existing_urls = {_normalized_url(item.url) for item in state.evidence}
    deduped_hits: list[SearchHit] = []
    seen_urls = set(existing_urls)
    for hit in hits:
        normalized = _normalized_url(hit.url)
        if not normalized or normalized in seen_urls:
            continue
        if not is_safe_citation_url(hit.url):
            logger.debug("[retriever] Dropping unsafe citation URL: %s", hit.url)
            continue
        seen_urls.add(normalized)
        deduped_hits.append(hit)

    if not deduped_hits:
        return {"evidence": [], "estimated_evidence_tokens": state.estimated_evidence_tokens}

    ranked_hits = heuristic_prefilter_hits(
        state.claim_text,
        deduped_hits,
        top_n=RANKER_HEURISTIC_TOP_N,
        evaluation_frame=extract_evaluation_frame(state.claim_text),
    )

    settings = get_settings()
    top_hits_for_fetch = ranked_hits[:FULL_PAGE_FETCH_TOP_N]
    fetch_tasks = [_resolve_page_text(hit, settings) for hit, _ in top_hits_for_fetch]
    fetched_texts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    fetched_by_url: dict[str, str] = {}
    for (hit, _), result in zip(top_hits_for_fetch, fetched_texts):
        if isinstance(result, str) and result:
            fetched_by_url[hit.url] = result
            logger.debug("[retriever] Fetched %d chars from %s", len(result), hit.url)

    new_evidence: list[EvidenceItem] = []
    new_tokens = 0
    for hit, score in ranked_hits:
        if hit.url in fetched_by_url:
            raw_text = fetched_by_url[hit.url]
            snippet = _truncate_snippet(raw_text, max_words=MAX_SNIPPET_WORDS * 2)
            content_source = "fetched"
        else:
            snippet = _truncate_snippet(hit.snippet)
            content_source = "snippet"

        source_index = len(state.evidence) + len(new_evidence) + 1
        token_count = estimate_formatted_evidence_tokens(
            url=hit.url,
            title=hit.title,
            snippet=snippet,
            source_index=source_index,
        )
        if state.estimated_evidence_tokens + new_tokens + token_count > state.max_evidence_tokens:
            break

        new_tokens += token_count
        new_evidence.append(
            EvidenceItem(
                url=hit.url,
                title=hit.title,
                snippet=snippet,
                content_source=content_source,
                credibility_tier=classify_domain(hit.url),
                relevance_score=score,
            )
        )

    logger.debug(
        "[retriever] Added %d evidence items (%d fetched, %d snippet-only)",
        len(new_evidence),
        sum(1 for item in new_evidence if item.content_source == "fetched"),
        sum(1 for item in new_evidence if item.content_source == "snippet"),
    )

    return {
        "evidence": new_evidence,
        "estimated_evidence_tokens": state.estimated_evidence_tokens + new_tokens,
    }
