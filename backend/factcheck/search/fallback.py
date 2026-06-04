"""Fallback search orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from factcheck.config import AppSettings, get_settings
from factcheck.search.models import SearchHit
from factcheck.search.providers import SearchProvider, build_provider_chain


logger = logging.getLogger(__name__)


async def search_with_fallback(
    query: str,
    *,
    max_results: int | None = None,
    settings: AppSettings | None = None,
    providers: Sequence[SearchProvider] | None = None,
) -> tuple[list[SearchHit], str | None]:
    """Try search providers in order until one returns hits."""

    resolved_settings = settings or get_settings()
    resolved_max_results = max_results or resolved_settings.search_max_results
    provider_chain = list(providers) if providers is not None else build_provider_chain(resolved_settings)

    for provider in provider_chain:
        try:
            hits = await provider.search(query, resolved_max_results)
        except Exception as exc:
            logger.warning("%s search failed for %r: %s", provider.name, query, exc)
            continue

        if hits:
            return hits, provider.name

        logger.info("%s returned no results for %r", provider.name, query)

    logger.warning("All search providers failed or returned no results for %r", query)
    return [], None
