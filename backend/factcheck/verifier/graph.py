"""LangGraph wrapper for the verifier pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factcheck.verifier.nodes.evidence_evaluator import evidence_evaluator_node
from factcheck.verifier.nodes.query_generator import query_generator_node
from factcheck.verifier.nodes.retriever import retriever_node
from factcheck.verifier.schemas import VerifierState


def route_after_query(state: VerifierState) -> str:
    """Route to retrieval when queries exist, otherwise evaluate existing evidence."""

    if state.current_queries or state.current_query:
        return "retriever"
    return "evidence_evaluator"


def route_after_evaluate(state: VerifierState) -> str:
    """Loop for more evidence until a final claim result exists or the cap is reached."""

    if state.claim_result is not None:
        return END

    if (
        state.intermediate_assessment is not None
        and state.intermediate_assessment.needs_more_evidence
        and state.iteration_count < state.max_iterations
    ):
        return "query_generator"

    return END


def build_verifier_graph():
    """Build the iterative verifier subgraph."""

    graph = StateGraph(VerifierState)
    graph.add_node("query_generator", query_generator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)

    graph.add_edge(START, "query_generator")
    graph.add_conditional_edges(
        "query_generator",
        route_after_query,
        {"retriever": "retriever", "evidence_evaluator": "evidence_evaluator"},
    )
    graph.add_edge("retriever", "evidence_evaluator")
    graph.add_conditional_edges(
        "evidence_evaluator",
        route_after_evaluate,
        {"query_generator": "query_generator", END: END},
    )

    return graph.compile()
