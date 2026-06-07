"""Phase 1 LangGraph skeleton for the fact-checking pipeline."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from factcheck.agents.dialogue import dialogue_node
from factcheck.agents.extractor import extractor_node
from factcheck.agents.orchestrator import orchestrator_node
from factcheck.agents.reporter import reporter_node
from factcheck.agents.verifier import verifier_node
from factcheck.state import FactCheckState


VerifierRoute = Literal["verifier", "reporter", "end"]


def _route_after_verifier(state: FactCheckState) -> VerifierRoute:
    if state["status"] == "error" or state["error"]:
        return "end"
    if len(state["claim_results"]) < len(state["extracted_claims"]):
        return "verifier"
    return "reporter"


def build_graph():
    """Build the stub pipeline graph from the Phase 1 architecture."""

    graph = StateGraph(FactCheckState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("dialogue", dialogue_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "extractor")
    graph.add_edge("extractor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"verifier": "verifier", "reporter": "reporter", "end": END},
    )
    graph.add_edge("reporter", END)
    graph.add_edge("dialogue", END)

    return graph.compile()
