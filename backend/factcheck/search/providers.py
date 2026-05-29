"""Search provider implementations."""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx

from factcheck.config import AppSettings
from factcheck.search.models import SearchHit


class SearchProvider(Protocol):
    """Provider interface for search backends."""

    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Run a web search and return normalized hits."""


class DuckDuckGoProvider:
    """DuckDuckGo search provider."""

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        def _search() -> list[SearchHit]:
            from duckduckgo_search import DDGS

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

        return await asyncio.to_thread(_search)


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
