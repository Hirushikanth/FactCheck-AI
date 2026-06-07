from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.verifier.nodes import evidence_ranker
from factcheck.verifier.nodes.evidence_ranker import (
    EvidenceRanking,
    RankerOutput,
    evidence_ranker_node,
    heuristic_prefilter_hits,
)
from factcheck.verifier.prompts import EVIDENCE_RANKER_SYSTEM_PROMPT
from factcheck.verifier.schemas import VerifierState


def test_evidence_ranker_prompt_documents_structured_fields() -> None:
    prompt = EVIDENCE_RANKER_SYSTEM_PROMPT.lower()

    assert "rankings" in prompt
    assert "index" in prompt
    assert "relevance_score" in prompt
    assert "zero-based" in prompt


def test_heuristic_prefilter_drops_empty_and_near_duplicate_snippets() -> None:
    hits = [
        SearchHit(
            url="https://nasa.gov/earth-shape",
            title="Earth shape",
            snippet="Earth is an oblate spheroid with a rounded shape.",
        ),
        SearchHit(
            url="https://nasa.gov/earth-shape-copy",
            title="Earth shape copy",
            snippet="Earth is an oblate spheroid with a rounded shape.",
        ),
        SearchHit(
            url="https://example.com/empty",
            title="No snippet",
            snippet="",
        ),
        SearchHit(
            url="https://space.example/mars",
            title="Mars facts",
            snippet="Mars has two moons named Phobos and Deimos.",
        ),
    ]

    filtered = heuristic_prefilter_hits(
        "The Earth is an oblate spheroid.",
        hits,
        top_n=3,
    )

    assert [hit.url for hit in filtered] == [
        "https://nasa.gov/earth-shape",
        "https://space.example/mars",
    ]


async def test_evidence_ranker_uses_llm_scores_to_select_evidence(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return RankerOutput(
            rankings=[
                EvidenceRanking(index=0, relevance_score=0.95, rationale="Direct"),
                EvidenceRanking(index=1, relevance_score=0.1, rationale="Tangential"),
                EvidenceRanking(index=99, relevance_score=1.0, rationale="Invalid"),
            ]
        )

    monkeypatch.setattr(evidence_ranker, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(evidence_ranker, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_ranker_node(
        VerifierState(
            claim="The Earth is an oblate spheroid.",
            raw_hits=[
                SearchHit(
                    url="https://generic.example/earth",
                    title="Earth",
                    snippet="Earth is the third planet from the Sun.",
                ),
                SearchHit(
                    url="https://science.example/earth-shape",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                ),
            ],
        )
    )

    assert [item.url for item in result["ranked_evidence"]] == [
        "https://science.example/earth-shape"
    ]


async def test_evidence_ranker_normalizes_one_based_llm_indexes(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return RankerOutput(
            rankings=[
                EvidenceRanking(index=1, relevance_score=0.95, rationale="Direct"),
                EvidenceRanking(index=2, relevance_score=0.1, rationale="Tangential"),
            ]
        )

    monkeypatch.setattr(evidence_ranker, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(evidence_ranker, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_ranker_node(
        VerifierState(
            claim="Water boils at 100 degrees Celsius at sea level.",
            raw_hits=[
                SearchHit(
                    url="https://science.example/boiling-point",
                    title="Boiling point of water",
                    snippet="Water boils at 100 degrees Celsius at sea level.",
                ),
                SearchHit(
                    url="https://weather.example/sea-level",
                    title="Sea level pressure",
                    snippet="Sea level pressure affects weather patterns.",
                ),
            ],
        )
    )

    assert [item.url for item in result["ranked_evidence"]] == [
        "https://science.example/boiling-point"
    ]


async def test_evidence_ranker_uses_heuristic_fallback_for_strong_hits(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return None

    monkeypatch.setattr(evidence_ranker, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(evidence_ranker, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_ranker_node(
        VerifierState(
            claim="Water boils at 100 degrees Celsius at sea level.",
            search_queries=["water boils 100 degrees Celsius sea level"],
            raw_hits=[
                SearchHit(
                    url="https://science.example/boiling-point",
                    title="Boiling point of water",
                    snippet="Water boils at 100 degrees Celsius at sea level.",
                )
            ],
        )
    )

    assert [item.url for item in result["ranked_evidence"]] == [
        "https://science.example/boiling-point"
    ]


async def test_evidence_ranker_still_fails_for_weak_hits_when_llm_ranking_fails(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return RankerOutput(rankings=[])

    monkeypatch.setattr(evidence_ranker, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(evidence_ranker, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_ranker_node(
        VerifierState(
            claim="The Earth is an oblate spheroid.",
            search_queries=["Earth oblate spheroid"],
            raw_hits=[
                SearchHit(
                    url="https://space.example/mars",
                    title="Mars facts",
                    snippet="Mars has two moons named Phobos and Deimos.",
                )
            ],
        )
    )

    assert result["ranked_evidence"] == []
    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["reasoning"] == "The evidence ranker did not return usable scores."
