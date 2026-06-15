from __future__ import annotations

from factcheck.state import ClaimResult
from factcheck.verifier import graph as verifier_graph
from factcheck.verifier import run_verifier
from factcheck.verifier.graph import build_verifier_graph
from factcheck.verifier.schemas import EvidenceItem, IntermediateAssessment, VerifierState


async def test_verifier_graph_retries_when_evaluator_requests_more_evidence(monkeypatch) -> None:
    calls: list[str] = []

    async def query_generator_node(state):
        calls.append("query_generator")
        query = "earth shape official source" if state.iteration_count else "earth shape"
        return {"current_query": query, "all_queries": state.all_queries + [query]}

    async def retriever_node(state):
        calls.append("retriever")
        return {
            "evidence": [
                EvidenceItem(
                    url="https://example.com/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                    relevance_score=0.95,
                )
            ],
            "estimated_evidence_tokens": state.estimated_evidence_tokens + 8,
        }

    async def evidence_evaluator_node(state):
        calls.append("evidence_evaluator")
        if state.iteration_count == 0:
            return {
                "intermediate_assessment": IntermediateAssessment(
                    needs_more_evidence=True,
                    missing_aspects=["official source"],
                ),
                "iteration_count": 1,
            }
        return {
            "claim_result": {
                "claim": state.claim_text,
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence": [item.snippet for item in state.evidence],
                "sources": [item.url for item in state.evidence],
                "reasoning": "The evidence directly supports the claim.",
                "search_queries": state.all_queries,
            }
        }

    monkeypatch.setattr(verifier_graph, "query_generator_node", query_generator_node)
    monkeypatch.setattr(verifier_graph, "retriever_node", retriever_node)
    monkeypatch.setattr(verifier_graph, "evidence_evaluator_node", evidence_evaluator_node)

    graph = build_verifier_graph()
    result = await graph.ainvoke(VerifierState(claim_text="The Earth is an oblate spheroid."))

    assert calls == [
        "query_generator",
        "retriever",
        "evidence_evaluator",
        "query_generator",
        "retriever",
        "evidence_evaluator",
    ]
    assert result["claim_result"]["verdict"] == "SUPPORTED"
    assert result["claim_result"]["search_queries"] == [
        "earth shape",
        "earth shape official source",
    ]


async def test_verifier_graph_routes_to_evaluator_when_no_query_is_generated(monkeypatch) -> None:
    calls: list[str] = []

    async def query_generator_node(state):
        calls.append("query_generator")
        return {"current_query": None, "search_exhausted": True}

    async def retriever_node(state):
        calls.append("retriever")
        return {"evidence": []}

    async def evidence_evaluator_node(state):
        calls.append("evidence_evaluator")
        return {
            "claim_result": {
                "claim": state.claim_text,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "evidence": [],
                "sources": [],
                "reasoning": "No search query could be generated.",
                "search_queries": state.all_queries,
            }
        }

    monkeypatch.setattr(verifier_graph, "query_generator_node", query_generator_node)
    monkeypatch.setattr(verifier_graph, "retriever_node", retriever_node)
    monkeypatch.setattr(verifier_graph, "evidence_evaluator_node", evidence_evaluator_node)

    graph = build_verifier_graph()
    result = await graph.ainvoke(VerifierState(claim_text="No searchable query."))

    assert calls == ["query_generator", "evidence_evaluator"]
    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"


async def test_verifier_graph_routes_to_evaluator_on_duplicate_query(monkeypatch) -> None:
    calls: list[str] = []

    async def query_generator_node(state):
        calls.append("query_generator")
        if state.iteration_count == 0:
            return {
                "current_query": "earth shape",
                "all_queries": state.all_queries + ["earth shape"],
            }
        return {"current_query": None, "search_exhausted": True}

    async def retriever_node(state):
        calls.append("retriever")
        return {
            "evidence": [
                EvidenceItem(
                    url="https://example.com/earth",
                    title="Earth shape",
                    snippet="Earth is an oblate spheroid.",
                    relevance_score=0.95,
                )
            ],
            "estimated_evidence_tokens": state.estimated_evidence_tokens + 8,
        }

    async def evidence_evaluator_node(state):
        calls.append("evidence_evaluator")
        if state.iteration_count == 0:
            return {
                "intermediate_assessment": IntermediateAssessment(
                    needs_more_evidence=True,
                    missing_aspects=["official source"],
                ),
                "iteration_count": 1,
            }
        return {
            "claim_result": {
                "claim": state.claim_text,
                "verdict": "SUPPORTED",
                "confidence": 0.85,
                "evidence": [item.snippet for item in state.evidence],
                "sources": [item.url for item in state.evidence],
                "reasoning": "Existing evidence supports the claim.",
                "search_queries": state.all_queries,
            }
        }

    monkeypatch.setattr(verifier_graph, "query_generator_node", query_generator_node)
    monkeypatch.setattr(verifier_graph, "retriever_node", retriever_node)
    monkeypatch.setattr(verifier_graph, "evidence_evaluator_node", evidence_evaluator_node)

    graph = build_verifier_graph()
    result = await graph.ainvoke(VerifierState(claim_text="The Earth is an oblate spheroid."))

    assert calls == [
        "query_generator",
        "retriever",
        "evidence_evaluator",
        "query_generator",
        "evidence_evaluator",
    ]
    assert result["claim_result"]["verdict"] == "SUPPORTED"
    assert result["claim_result"]["search_queries"] == ["earth shape"]


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
            assert state.claim_text == "The Earth is an oblate spheroid."
            return {"claim_result": expected}

    monkeypatch.setattr(verifier_graph, "build_verifier_graph", lambda: FakeVerifierGraph())

    assert await run_verifier("The Earth is an oblate spheroid.") == expected
