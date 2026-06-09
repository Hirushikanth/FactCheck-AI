from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.verifier.nodes import retriever
from factcheck.verifier.nodes.retriever import _estimate_tokens, _truncate_snippet, retriever_node
from factcheck.verifier.schemas import EvidenceItem, VerifierState


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


def test_estimate_tokens_uses_conservative_word_ratio() -> None:
    assert _estimate_tokens("one two three") == 4


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
