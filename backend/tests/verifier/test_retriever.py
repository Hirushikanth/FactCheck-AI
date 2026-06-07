from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.state import ClaimResult
from factcheck.verifier.nodes import retriever
from factcheck.verifier.nodes.retriever import retriever_node
from factcheck.verifier.schemas import VerifierState


async def test_retriever_searches_each_query_and_dedupes_by_url(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_search_with_fallback(query: str):
        calls.append(query)
        if query == "earth shape":
            return (
                [
                    SearchHit(
                        url="https://example.com/earth/",
                        title="Earth",
                        snippet="Earth is an oblate spheroid.",
                    )
                ],
                "duckduckgo",
            )
        return (
            [
                SearchHit(
                    url="https://example.com/earth",
                    title="Duplicate",
                    snippet="Earth has a rounded shape.",
                ),
                SearchHit(
                    url="https://space.example/earth",
                    title="Earth facts",
                    snippet="Earth is the third planet from the Sun.",
                ),
            ],
            "duckduckgo",
        )

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim="The Earth is an oblate spheroid.",
            search_queries=["earth shape", "earth oblate spheroid"],
        )
    )

    assert calls == ["earth shape", "earth oblate spheroid"]
    assert [hit.url for hit in result["raw_hits"]] == [
        "https://example.com/earth/",
        "https://space.example/earth",
    ]


async def test_retriever_sets_insufficient_evidence_when_search_returns_no_hits(
    monkeypatch,
) -> None:
    async def fake_search_with_fallback(query: str):
        return [], None

    monkeypatch.setattr(retriever, "search_with_fallback", fake_search_with_fallback)

    result = await retriever_node(
        VerifierState(
            claim="A claim with no web evidence.",
            search_queries=["claim no evidence"],
        )
    )

    claim_result: ClaimResult = result["claim_result"]
    assert result["raw_hits"] == []
    assert claim_result["claim"] == "A claim with no web evidence."
    assert claim_result["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert claim_result["confidence"] == 0.0
    assert claim_result["evidence"] == []
    assert claim_result["sources"] == []
    assert claim_result["search_queries"] == ["claim no evidence"]
