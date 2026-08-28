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

# Global DDG request state protected by an asyncio lock
_last_ddg_request_at: float | None = None
_ddg_lock: asyncio.Lock | None = None


def _get_ddg_lock() -> asyncio.Lock:
    """Return the global DDG lock, creating it lazily on the running event loop."""
    global _ddg_lock
    if _ddg_lock is None:
        _ddg_lock = asyncio.Lock()
    return _ddg_lock


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


async def _post_json_with_retry(
    *,
    provider_name: str,
    url: str,
    headers: dict[str, str] | None,
    payload: dict[str, object],
    settings: AppSettings,
) -> dict[str, object]:
    """POST JSON with bounded retries for transient provider failures."""

    for attempt in range(settings.search_api_max_retries):
        retryable_error: Exception | None = None
        try:
            async with httpx.AsyncClient(timeout=settings.search_api_timeout_seconds) as client:
                request_kwargs: dict[str, object] = {"json": payload}
                if headers is not None:
                    request_kwargs["headers"] = headers
                response = await client.post(url, **request_kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            retryable_error = exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 429 and (status_code is None or not 500 <= status_code <= 599):
                raise
            retryable_error = exc

        if attempt + 1 >= settings.search_api_max_retries:
            assert retryable_error is not None
            raise retryable_error

        delay = _jittered_backoff(
            attempt,
            settings.search_api_retry_base_delay,
            settings.search_api_retry_max_delay,
        )
        logger.warning(
            "%s search request failed (attempt %d/%d); retrying in %.1fs",
            provider_name,
            attempt + 1,
            settings.search_api_max_retries,
            delay,
        )
        await asyncio.sleep(delay)

    raise RuntimeError(f"{provider_name} retry loop exited unexpectedly")


async def _enforce_ddg_spacing(settings: AppSettings) -> None:
    """Enforce minimum spacing between DDG requests using a global async lock."""

    global _last_ddg_request_at

    if settings.ddg_min_request_interval <= 0:
        return

    lock = _get_ddg_lock()

    async with lock:
        if _last_ddg_request_at is not None:
            elapsed = time.monotonic() - _last_ddg_request_at
            if elapsed < settings.ddg_min_request_interval:
                wait = (settings.ddg_min_request_interval - elapsed) + random.uniform(0, 0.5)
                logger.debug(
                    "[ddg] Rate limit spacing: waiting %.2fs before next request",
                    wait,
                )
                await asyncio.sleep(wait)

        _last_ddg_request_at = time.monotonic()


def reset_ddg_spacing_for_tests() -> None:
    """Reset cached spacing state so tests do not inherit timing from prior runs."""

    global _last_ddg_request_at, _ddg_lock
    _last_ddg_request_at = None
    _ddg_lock = None


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
        settings = get_settings()
        payload = await _post_json_with_retry(
            provider_name=self.name,
            url="https://api.tavily.com/search",
            headers=None,
            payload={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": True,
            },
            settings=settings,
        )

        hits: list[SearchHit] = []
        for result in payload.get("results", []):
            if not isinstance(result, dict) or not result.get("url"):
                continue
            raw_content = str(result.get("raw_content", "")).strip()
            hits.append(
                SearchHit(
                    url=str(result.get("url", "")),
                    title=str(result.get("title", "")),
                    snippet=str(result.get("content", "")),
                    page_text=raw_content or None,
                )
            )
        return hits


class SerperProvider:
    """Serper Google Search API provider."""

    name = "serper"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        settings = get_settings()
        payload = await _post_json_with_retry(
            provider_name=self.name,
            url="https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            payload={"q": query, "num": max_results},
            settings=settings,
        )

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
