"""LangGraph wrapper for the verifier pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factcheck.verifier.nodes.evidence_ranker import evidence_ranker_node
from factcheck.verifier.nodes.query_generator import query_generator_node
from factcheck.verifier.nodes.retriever import retriever_node
from factcheck.verifier.nodes.verdict_engine import verdict_engine_node
from factcheck.verifier.schemas import VerifierState


def build_verifier_graph():
    """Build the sequential verifier subgraph."""

    graph = StateGraph(VerifierState)
    graph.add_node("query_generator", query_generator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("evidence_ranker", evidence_ranker_node)
    graph.add_node("verdict_engine", verdict_engine_node)

    graph.add_edge(START, "query_generator")
    graph.add_edge("query_generator", "retriever")
    graph.add_edge("retriever", "evidence_ranker")
    graph.add_edge("evidence_ranker", "verdict_engine")
    graph.add_edge("verdict_engine", END)

    return graph.compile()
