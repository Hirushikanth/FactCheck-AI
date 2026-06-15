"""LangGraph pipeline for the fact-checking workflow.

Pipeline: orchestrator → extractor → verifier → reporter → END

The dialogue agent is NOT part of this pipeline.  It runs as a standalone
graph on demand when a follow-up message arrives for a completed session.
See ``factcheck.dialogue`` for the dialogue agent entry point.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factcheck.agents.extractor import extractor_node
from factcheck.agents.orchestrator import orchestrator_node
from factcheck.agents.reporter import reporter_node
from factcheck.agents.verifier import verifier_node
from factcheck.state import FactCheckState


def build_graph():
    """Build and compile the fact-check pipeline graph."""

    graph = StateGraph(FactCheckState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "extractor")
    graph.add_edge("extractor", "verifier")
    graph.add_edge("verifier", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


def _initial_state(*, session_id: str, text: str) -> FactCheckState:
    return {
        "raw_input": text,
        "extracted_claims": [],
        "claim_results": [],
        "final_report": None,
        "messages": [],
        "current_agent": "",
        "session_id": session_id,
        "error": None,
        "status": "idle",
    }


async def run_factcheck_pipeline(
    *,
    session_id: str,
    text: str,
) -> FactCheckState:
    """Run the full fact-check pipeline for *text* and return the final state."""

    graph = build_graph()
    result: FactCheckState = await graph.ainvoke(_initial_state(session_id=session_id, text=text))
    return result
