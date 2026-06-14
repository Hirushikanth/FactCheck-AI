"""Search provider implementations."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Protocol

import httpx

from factcheck.config import AppSettings, get_settings
from factcheck.search.models import SearchHit


logger = logging.getLogger(__name__)

_last_ddg_request_at: float | None = None


def _jittered_backoff(attempt: int, base: float, max_delay: float) -> float:
    """Return a randomized exponential backoff delay in seconds."""

    cap = min(base * (2**attempt), max_delay)
    return random.uniform(0, cap)


def _is_likely_throttle(message: str) -> bool:
    """Heuristic check for DuckDuckGo rate-limit responses in error text."""

    lowered = message.lower()
    return any(token in lowered for token in ("403", "202", "rate", "ratelimit"))


def _truncate_query(query: str, max_len: int = 80) -> str:
    if len(query) <= max_len:
        return query
    return query[: max_len - 3] + "..."


async def _enforce_ddg_spacing(settings: AppSettings) -> None:
    """Sleep if the previous DuckDuckGo request was too recent."""

    global _last_ddg_request_at

    if settings.ddg_min_request_interval <= 0 or _last_ddg_request_at is None:
        return

    elapsed = time.monotonic() - _last_ddg_request_at
    if elapsed >= settings.ddg_min_request_interval:
        return

    wait = (settings.ddg_min_request_interval - elapsed) + random.uniform(0, 0.5)
    logger.warning(
        "Spacing DuckDuckGo requests to avoid rate limits; waiting %.1fs",
        wait,
    )
    await asyncio.sleep(wait)


def _record_ddg_request() -> None:
    global _last_ddg_request_at
    _last_ddg_request_at = time.monotonic()


def reset_ddg_spacing_for_tests() -> None:
    """Reset cached spacing state so tests do not inherit timing from prior runs."""

    global _last_ddg_request_at
    _last_ddg_request_at = None


class SearchProvider(Protocol):
    """Provider interface for search backends."""

    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Run a web search and return normalized hits."""


class DuckDuckGoProvider:
    """DuckDuckGo search provider."""

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        settings = get_settings()
        short_query = _truncate_query(query)
        max_retries = settings.ddg_max_retries

        def _search() -> list[SearchHit]:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [
                    SearchHit(
                        url=str(result.get("href", "")),
                        title=str(result.get("title", "")),
                        snippet=str(result.get("body", "")),
                    )
                    for result in results
                    if result.get("href")
                ]

        for attempt in range(max_retries):
            await _enforce_ddg_spacing(settings)

            try:
                hits = await asyncio.to_thread(_search)
            except Exception as exc:
                _record_ddg_request()
                throttle_note = " (likely rate limited)" if _is_likely_throttle(str(exc)) else ""
                is_last = attempt + 1 >= max_retries
                logger.warning(
                    "DuckDuckGo search failed for %r (attempt %d/%d)%s: %s",
                    short_query,
                    attempt + 1,
                    max_retries,
                    throttle_note,
                    exc,
                )
                if is_last:
                    return []
                delay = _jittered_backoff(
                    attempt,
                    settings.ddg_retry_base_delay,
                    settings.ddg_retry_max_delay,
                )
                logger.warning("DuckDuckGo retrying in %.1fs", delay)
                await asyncio.sleep(delay)
                continue

            _record_ddg_request()

            if hits:
                return hits

            is_last = attempt + 1 >= max_retries
            logger.warning(
                "DuckDuckGo returned no results for %r (attempt %d/%d); possible rate limit",
                short_query,
                attempt + 1,
                max_retries,
            )
            if is_last:
                return []

            delay = _jittered_backoff(
                attempt,
                settings.ddg_retry_base_delay,
                settings.ddg_retry_max_delay,
            )
            logger.warning("DuckDuckGo retrying in %.1fs", delay)
            await asyncio.sleep(delay)

        return []


class TavilyProvider:
    """Tavily Search API provider."""

    name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            payload = response.json()

        return [
            SearchHit(
                url=str(result.get("url", "")),
                title=str(result.get("title", "")),
                snippet=str(result.get("content", "")),
            )
            for result in payload.get("results", [])
            if isinstance(result, dict) and result.get("url")
        ]


class SerperProvider:
    """Serper Google Search API provider."""

    name = "serper"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
            )
            response.raise_for_status()
            payload = response.json()

        return [
            SearchHit(
                url=str(result.get("link", "")),
                title=str(result.get("title", "")),
                snippet=str(result.get("snippet", "")),
            )
            for result in payload.get("organic", [])
            if isinstance(result, dict) and result.get("link")
        ]


def build_provider_chain(settings: AppSettings) -> list[SearchProvider]:
    """Build configured providers in fallback order, skipping missing API keys."""

    providers: list[SearchProvider] = []
    for provider_name in (item.strip().lower() for item in settings.search_provider_order.split(",")):
        if provider_name == "duckduckgo":
            providers.append(DuckDuckGoProvider())
        elif provider_name == "tavily" and settings.tavily_api_key:
            providers.append(TavilyProvider(settings.tavily_api_key))
        elif provider_name == "serper" and settings.serper_api_key:
            providers.append(SerperProvider(settings.serper_api_key))

    return providers
