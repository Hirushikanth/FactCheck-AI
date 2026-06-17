from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from factcheck.config import AppSettings, get_settings
from factcheck.search import SearchHit, build_provider_chain, search_with_fallback
from factcheck.search.providers import (
    DuckDuckGoProvider,
    TavilyProvider,
    _jittered_backoff,
    _is_likely_throttle,
    reset_ddg_spacing_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_ddg_provider_state() -> None:
    reset_ddg_spacing_for_tests()
    get_settings.cache_clear()
    yield
    reset_ddg_spacing_for_tests()
    get_settings.cache_clear()


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


def test_is_likely_throttle_detects_common_signals() -> None:
    assert _is_likely_throttle("HTTP 403 Forbidden")
    assert _is_likely_throttle("received 202")
    assert _is_likely_throttle("Ratelimit exceeded")
    assert not _is_likely_throttle("connection reset")


def test_jittered_backoff_stays_within_cap(monkeypatch) -> None:
    monkeypatch.setattr("factcheck.search.providers.random.uniform", lambda low, high: high)

    assert _jittered_backoff(0, base=1.0, max_delay=8.0) == 1.0
    assert _jittered_backoff(3, base=1.0, max_delay=8.0) == 8.0


def _install_fake_ddgs(monkeypatch, fake_ddgs_cls: type) -> None:
    fake_ddgs_module = types.SimpleNamespace(DDGS=fake_ddgs_cls)
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs_module)


async def test_ddg_retries_on_exception_then_succeeds(monkeypatch) -> None:
    class FakeDDGS:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            FakeDDGS.calls += 1
            if FakeDDGS.calls == 1:
                raise RuntimeError("HTTP 403 Forbidden")
            return [
                {
                    "href": "https://example.com/result",
                    "title": "Result",
                    "body": "Evidence text.",
                }
            ]

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("DDG_MIN_REQUEST_INTERVAL", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "factcheck.search.providers._jittered_backoff",
        lambda attempt, base, max_delay: 0.5,
    )
    _install_fake_ddgs(monkeypatch, FakeDDGS)

    hits = await DuckDuckGoProvider().search("test query", max_results=1)

    assert FakeDDGS.calls == 2
    assert sleeps == [0.5]
    assert hits == [
        SearchHit(
            url="https://example.com/result",
            title="Result",
            snippet="Evidence text.",
        )
    ]


async def test_ddg_retries_on_empty_results(monkeypatch) -> None:
    class FakeDDGS:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            FakeDDGS.calls += 1
            if FakeDDGS.calls == 1:
                return []
            return [
                {
                    "href": "https://example.com/result",
                    "title": "Result",
                    "body": "Evidence text.",
                }
            ]

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("DDG_MIN_REQUEST_INTERVAL", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "factcheck.search.providers._jittered_backoff",
        lambda attempt, base, max_delay: 1.0,
    )
    _install_fake_ddgs(monkeypatch, FakeDDGS)

    hits = await DuckDuckGoProvider().search("empty then hit", max_results=1)

    assert FakeDDGS.calls == 2
    assert sleeps == [1.0]
    assert len(hits) == 1


async def test_ddg_returns_empty_after_max_retries(monkeypatch) -> None:
    class FakeDDGS:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            FakeDDGS.calls += 1
            raise RuntimeError("HTTP 403 Forbidden")

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv("DDG_MAX_RETRIES", "3")
    monkeypatch.setenv("DDG_MIN_REQUEST_INTERVAL", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "factcheck.search.providers._jittered_backoff",
        lambda attempt, base, max_delay: float(attempt + 1),
    )
    _install_fake_ddgs(monkeypatch, FakeDDGS)

    hits = await DuckDuckGoProvider().search("always fails", max_results=1)

    assert FakeDDGS.calls == 3
    assert hits == []
    assert sleeps == [1.0, 2.0]


async def test_ddg_backoff_sleeps_between_retries(monkeypatch) -> None:
    class FakeDDGS:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            FakeDDGS.calls += 1
            return []

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv("DDG_MAX_RETRIES", "3")
    monkeypatch.setenv("DDG_MIN_REQUEST_INTERVAL", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "factcheck.search.providers._jittered_backoff",
        lambda attempt, base, max_delay: float((attempt + 1) * 2),
    )
    _install_fake_ddgs(monkeypatch, FakeDDGS)

    hits = await DuckDuckGoProvider().search("always empty", max_results=1)

    assert hits == []
    assert FakeDDGS.calls == 3
    assert sleeps == [2.0, 4.0]


async def test_ddg_enforces_min_request_interval(monkeypatch) -> None:
    import factcheck.search.providers as providers_module

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, max_results: int):
            return [
                {
                    "href": "https://example.com/spaced",
                    "title": "Spaced",
                    "body": "Spaced result.",
                }
            ]

    providers_module._last_ddg_request_at = time.monotonic()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv("DDG_MIN_REQUEST_INTERVAL", "2.0")
    monkeypatch.setenv("DDG_MAX_RETRIES", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("factcheck.search.providers.random.uniform", lambda low, high: 0.0)
    _install_fake_ddgs(monkeypatch, FakeDDGS)

    hits = await DuckDuckGoProvider().search("spaced query", max_results=1)

    assert len(hits) == 1
    assert len(sleeps) == 1
    assert sleeps[0] >= 1.9


async def test_tavily_provider_maps_raw_content_to_page_text(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Article",
                        "content": "Snippet text.",
                        "raw_content": "Full page article text.",
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.last_json: dict | None = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url: str, json: dict):
            self.last_json = json
            return FakeResponse()

    fake_client = FakeAsyncClient()
    monkeypatch.setattr(
        "factcheck.search.providers.httpx.AsyncClient",
        lambda *args, **kwargs: fake_client,
    )

    hits = await TavilyProvider("tvly-test").search("earth shape", max_results=1)

    assert fake_client.last_json is not None
    assert fake_client.last_json["include_raw_content"] is True
    assert hits == [
        SearchHit(
            url="https://example.com/article",
            title="Article",
            snippet="Snippet text.",
            page_text="Full page article text.",
        )
    ]
