from __future__ import annotations

from factcheck.verifier.nodes import verdict_engine
from factcheck.verifier.nodes.verdict_engine import VerdictOutput, verdict_engine_node
from factcheck.verifier.schemas import EvidenceItem, VerifierState


async def test_verdict_engine_maps_structured_output_to_claim_result(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return VerdictOutput(
            verdict="SUPPORTED",
            confidence=0.91,
            reasoning="The evidence directly supports the claim.",
        )

    monkeypatch.setattr(verdict_engine, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(verdict_engine, "call_llm_with_structured_output", fake_structured_call)

    result = await verdict_engine_node(
        VerifierState(
            claim="The Earth is an oblate spheroid.",
            search_queries=["Earth oblate spheroid"],
            ranked_evidence=[
                EvidenceItem(
                    url="https://science.example/earth-shape",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                    relevance_score=0.95,
                )
            ],
        )
    )

    assert result["claim_result"] == {
        "claim": "The Earth is an oblate spheroid.",
        "verdict": "SUPPORTED",
        "confidence": 0.91,
        "evidence": ["Earth is an oblate spheroid."],
        "sources": ["https://science.example/earth-shape"],
        "reasoning": "The evidence directly supports the claim.",
        "search_queries": ["Earth oblate spheroid"],
    }


async def test_verdict_engine_returns_insufficient_evidence_without_ranked_evidence() -> None:
    result = await verdict_engine_node(
        VerifierState(
            claim="A claim with no relevant evidence.",
            search_queries=["claim evidence"],
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["confidence"] == 0.0
    assert result["claim_result"]["evidence"] == []
    assert result["claim_result"]["sources"] == []
