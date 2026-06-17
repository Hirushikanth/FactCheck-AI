from __future__ import annotations

from factcheck.config import AppSettings, get_settings
from factcheck.search import SearchHit
from factcheck.verifier.nodes import retriever
from factcheck.verifier.nodes.retriever import _truncate_snippet, retriever_node
from factcheck.verifier.schemas import EvidenceItem, VerifierState
from factcheck.verifier.utils import (
    estimate_formatted_evidence_tokens,
    estimate_tokens,
    heuristic_prefilter_hits,
)

_BERRIES_CLAIM = (
    "Strawberries are not berries [according to botanical definitions of fruits]"
)


async def test_retriever_searches_current_query_and_appends_new_evidence(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        calls.append(query)
        return (
            [
                SearchHit(
                    url="https://example.com/earth/",
                    title="Duplicate existing",
                    snippet="Earth is an oblate spheroid.",
                ),
                SearchHit(
                    url="https://space.example/earth",
                    title="Earth facts",
                    snippet="Earth is an oblate spheroid with a slightly flattened shape.",
                ),
            ],
            "duckduckgo",
        )

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
            all_queries=["earth oblate spheroid"],
            evidence=[
                EvidenceItem(
                    url="https://example.com/earth",
                    title="Existing",
                    snippet="Earlier source.",
                )
            ],
        )
    )

    assert calls == ["earth oblate spheroid"]
    assert [item.url for item in result["evidence"]] == [
        "https://space.example/earth",
    ]
    assert result["estimated_evidence_tokens"] > 0


async def test_retriever_returns_empty_evidence_when_search_returns_no_hits(
    monkeypatch,
) -> None:
    async def fake_search_with_fallback(query: str):
        return [], None

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim_text="A claim with no web evidence.",
            current_query="claim no evidence",
            all_queries=["claim no evidence"],
        )
    )

    assert result["evidence"] == []
    assert "claim_result" not in result


def test_truncate_snippet_respects_word_limit_and_sentence_boundary() -> None:
    snippet = "First sentence has useful evidence. " + " ".join(f"word{i}" for i in range(50))

    truncated = _truncate_snippet(snippet, max_words=6)

    assert truncated == "First sentence has useful evidence."


def test_estimate_tokens_uses_tiktoken() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") > 0


def test_estimate_tokens_counts_url_heavy_text_higher_than_word_split() -> None:
    url_heavy = "https://example.com/path/to/resource?query=value&other=12345"
    word_split_estimate = int(len(url_heavy.split()) / 0.75)

    assert estimate_tokens(url_heavy) > word_split_estimate


def test_estimate_formatted_evidence_tokens_includes_metadata_overhead() -> None:
    snippet = "Short evidence snippet."
    snippet_only = estimate_tokens(snippet)
    formatted = estimate_formatted_evidence_tokens(
        url="https://science.example/earth",
        title="Earth shape",
        snippet=snippet,
        source_index=1,
    )

    assert formatted > snippet_only


def test_heuristic_prefilter_ranks_botanical_above_colloquial_for_framed_claim() -> None:
    colloquial = SearchHit(
        url="https://example.com/popular",
        title="Berries",
        snippet="Strawberries are commonly called berries in everyday language.",
    )
    botanical = SearchHit(
        url="https://example.com/botany",
        title="Botanical berries",
        snippet="Botanically, strawberries are aggregate fruits, not true berries.",
    )

    ranked = heuristic_prefilter_hits(
        _BERRIES_CLAIM,
        [colloquial, botanical],
        top_n=2,
        evaluation_frame="according to botanical definitions of fruits",
    )

    assert [hit.url for hit, _score in ranked] == [
        "https://example.com/botany",
        "https://example.com/popular",
    ]


async def test_retriever_respects_evidence_token_budget(monkeypatch) -> None:
    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet=" ".join("evidence" for _ in range(40)),
                )
            ],
            "duckduckgo",
        )

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
            max_evidence_tokens=10,
        )
    )

    assert result["evidence"] == []
    assert result["estimated_evidence_tokens"] == 0


async def test_retriever_searches_all_current_queries_in_parallel(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        calls.append(query)
        return (
            [
                SearchHit(
                    url=f"https://example.com/{query.replace(' ', '-')}",
                    title=f"Results for {query}",
                    snippet=f"Evidence from {query}.",
                )
            ],
            "duckduckgo",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        return None

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_queries=["earth oblate spheroid official", "earth shape fact check"],
            all_queries=["earth oblate spheroid official", "earth shape fact check"],
        )
    )

    assert sorted(calls) == sorted(
        ["earth oblate spheroid official", "earth shape fact check"]
    )
    assert len(result["evidence"]) == 2


async def test_retriever_uses_fetched_page_text_for_top_hits(monkeypatch) -> None:
    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet="Short snippet only.",
                )
            ],
            "duckduckgo",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        return "Earth is an oblate spheroid with measurable flattening."

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert result["evidence"][0].snippet == (
        "Earth is an oblate spheroid with measurable flattening."
    )
    assert result["evidence"][0].content_source == "fetched"


async def test_retriever_falls_back_to_snippet_when_fetch_fails(monkeypatch) -> None:
    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                )
            ],
            "duckduckgo",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        return None

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert result["evidence"][0].snippet == "Earth is an oblate spheroid."
    assert result["evidence"][0].content_source == "snippet"


async def test_resolve_page_text_skips_blocked_domains() -> None:
    settings = AppSettings(full_page_fetch_mode="provider")
    assert (
        await retriever._resolve_page_text(
            SearchHit(
                url="https://twitter.com/post/123",
                snippet="Earth is round.",
            ),
            settings,
        )
        is None
    )


async def test_retriever_uses_snippet_for_blocked_domains(monkeypatch) -> None:
    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://twitter.com/post/123",
                    title="Tweet",
                    snippet="Earth is round.",
                )
            ],
            "duckduckgo",
        )

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth shape",
        )
    )

    assert result["evidence"][0].snippet == "Earth is round."
    assert result["evidence"][0].content_source == "snippet"


async def test_retriever_fetches_top_five_ranked_hits(monkeypatch) -> None:
    fetch_calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url=f"https://science.example/hit-{index}",
                    title=f"Hit {index}",
                    snippet=f"Snippet for hit {index}.",
                )
                for index in range(6)
            ],
            "duckduckgo",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        fetch_calls.append(url)
        return f"Full page text for {url}."

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert len(fetch_calls) == 5
    assert sum(item.content_source == "fetched" for item in result["evidence"]) == 5
    if len(result["evidence"]) > 5:
        assert result["evidence"][5].content_source == "snippet"


async def test_retriever_drops_unsafe_citation_urls(monkeypatch) -> None:
    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://127.0.0.1/internal",
                    title="Internal",
                    snippet="Should be dropped.",
                ),
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                ),
            ],
            "duckduckgo",
        )

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert [item.url for item in result["evidence"]] == ["https://science.example/earth"]


async def test_retriever_uses_provider_page_text_without_self_fetch(monkeypatch) -> None:
    fetch_calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet="Short snippet only.",
                    page_text="Earth is an oblate spheroid with measurable flattening.",
                )
            ],
            "tavily",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        fetch_calls.append(url)
        raise AssertionError("pinned fetch should not run when provider page_text exists")

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert fetch_calls == []
    assert result["evidence"][0].content_source == "fetched"
    assert "measurable flattening" in result["evidence"][0].snippet


async def test_retriever_off_mode_skips_pinned_fetch(monkeypatch) -> None:
    fetch_calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        return (
            [
                SearchHit(
                    url="https://science.example/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                )
            ],
            "duckduckgo",
        )

    async def fake_fetch_html_pinned(url: str, **kwargs) -> str:
        fetch_calls.append(url)
        raise AssertionError("pinned fetch should not run in off mode")

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)
    monkeypatch.setattr(retriever, "fetch_html_pinned", fake_fetch_html_pinned)
    monkeypatch.setattr(
        retriever,
        "get_settings",
        lambda: AppSettings(full_page_fetch_mode="off"),
    )
    get_settings.cache_clear()

    result = await retriever_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            current_query="earth oblate spheroid",
        )
    )

    assert fetch_calls == []
    assert result["evidence"][0].content_source == "snippet"
