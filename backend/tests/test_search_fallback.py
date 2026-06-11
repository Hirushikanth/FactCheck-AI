from __future__ import annotations

import sys
import types

from factcheck.config import AppSettings
from factcheck.search import SearchHit, build_provider_chain, search_with_fallback
from factcheck.search.providers import DuckDuckGoProvider


class FakeProvider:
    def __init__(self, name: str, hits: list[SearchHit] | None = None, raises: bool = False):
        self.name = name
        self.hits = hits or []
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls.append((query, max_results))
        if self.raises:
            raise RuntimeError(f"{self.name} unavailable")
        return self.hits


async def test_search_with_fallback_uses_next_provider_after_empty_results() -> None:
    duckduckgo = FakeProvider("duckduckgo", hits=[])
    tavily = FakeProvider(
        "tavily",
        hits=[SearchHit(url="https://example.com", title="Example", snippet="Evidence")],
    )
    serper = FakeProvider("serper", hits=[SearchHit(url="https://unused.com")])

    hits, provider_used = await search_with_fallback(
        "Ada Lovelace algorithm",
        max_results=2,
        providers=[duckduckgo, tavily, serper],
    )

    assert provider_used == "tavily"
    assert hits == [SearchHit(url="https://example.com", title="Example", snippet="Evidence")]
    assert duckduckgo.calls == [("Ada Lovelace algorithm", 2)]
    assert tavily.calls == [("Ada Lovelace algorithm", 2)]
    assert serper.calls == []


async def test_search_with_fallback_continues_after_provider_error() -> None:
    duckduckgo = FakeProvider("duckduckgo", raises=True)
    serper = FakeProvider("serper", hits=[SearchHit(url="https://serper.example")])

    hits, provider_used = await search_with_fallback(
        "query",
        max_results=1,
        providers=[duckduckgo, serper],
    )

    assert provider_used == "serper"
    assert hits == [SearchHit(url="https://serper.example")]


def test_build_provider_chain_skips_keyed_providers_without_keys() -> None:
    settings = AppSettings(
        search_provider_order="duckduckgo,tavily,serper",
        tavily_api_key=None,
        serper_api_key=None,
    )

    providers = build_provider_chain(settings)

    assert [provider.name for provider in providers] == ["duckduckgo"]


def test_build_provider_chain_includes_keyed_fallbacks_when_configured() -> None:
    settings = AppSettings(
        search_provider_order="duckduckgo,tavily,serper",
        tavily_api_key="tvly-test",
        serper_api_key="serper-test",
    )

    providers = build_provider_chain(settings)

    assert [provider.name for provider in providers] == ["duckduckgo", "tavily", "serper"]


async def test_duckduckgo_provider_uses_renamed_ddgs_package(monkeypatch) -> None:
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            assert query == "boiling point of water"
            assert max_results == 1
            return [
                {
                    "href": "https://example.com/water",
                    "title": "Water boiling point",
                    "body": "Water boils at 100 degrees Celsius at sea level.",
                }
            ]

    fake_ddgs_module = types.SimpleNamespace(DDGS=FakeDDGS)
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs_module)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

    hits = await DuckDuckGoProvider().search("boiling point of water", max_results=1)

    assert hits == [
        SearchHit(
            url="https://example.com/water",
            title="Water boiling point",
            snippet="Water boils at 100 degrees Celsius at sea level.",
        )
    ]
