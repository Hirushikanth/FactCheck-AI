from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.state import ClaimResult
from factcheck.verifier import graph as verifier_graph
from factcheck.verifier import run_verifier
from factcheck.verifier.graph import build_verifier_graph
from factcheck.verifier.schemas import EvidenceItem, VerifierState


async def test_verifier_graph_runs_stages_in_order(monkeypatch) -> None:
    calls: list[str] = []

    async def query_generator_node(state):
        calls.append("query_generator")
        return {"search_queries": ["earth oblate spheroid"]}

    async def retriever_node(state):
        calls.append("retriever")
        return {
            "raw_hits": [
                SearchHit(
                    url="https://example.com/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                )
            ]
        }

    async def evidence_ranker_node(state):
        calls.append("evidence_ranker")
        return {
            "ranked_evidence": [
                EvidenceItem(
                    url="https://example.com/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                    relevance_score=0.95,
                )
            ]
        }

    async def verdict_engine_node(state):
        calls.append("verdict_engine")
        return {
            "claim_result": {
                "claim": state.claim,
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence": ["Earth is an oblate spheroid."],
                "sources": ["https://example.com/earth"],
                "reasoning": "The evidence directly supports the claim.",
                "search_queries": state.search_queries,
            }
        }

    monkeypatch.setattr(verifier_graph, "query_generator_node", query_generator_node)
    monkeypatch.setattr(verifier_graph, "retriever_node", retriever_node)
    monkeypatch.setattr(verifier_graph, "evidence_ranker_node", evidence_ranker_node)
    monkeypatch.setattr(verifier_graph, "verdict_engine_node", verdict_engine_node)

    graph = build_verifier_graph()
    result = await graph.ainvoke(VerifierState(claim="The Earth is an oblate spheroid."))

    assert calls == ["query_generator", "retriever", "evidence_ranker", "verdict_engine"]
    assert result["claim_result"]["verdict"] == "SUPPORTED"


async def test_run_verifier_returns_claim_result(monkeypatch) -> None:
    expected: ClaimResult = {
        "claim": "The Earth is an oblate spheroid.",
        "verdict": "SUPPORTED",
        "confidence": 0.9,
        "evidence": ["Earth is an oblate spheroid."],
        "sources": ["https://example.com/earth"],
        "reasoning": "The evidence directly supports the claim.",
        "search_queries": ["earth oblate spheroid"],
    }

    class FakeVerifierGraph:
        async def ainvoke(self, state):
            assert state.claim == "The Earth is an oblate spheroid."
            return {"claim_result": expected}

    monkeypatch.setattr(verifier_graph, "build_verifier_graph", lambda: FakeVerifierGraph())

    assert await run_verifier("The Earth is an oblate spheroid.") == expected
