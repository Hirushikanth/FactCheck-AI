"""LangGraph wrapper for the reporter agent."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from factcheck.reporter.nodes.generate_report import generate_report_node
from factcheck.state import FactCheckState


def build_reporter_graph():
    """Build the single-node reporter graph."""

    graph = StateGraph(FactCheckState)
    graph.add_node("generate_report", generate_report_node)
    graph.set_entry_point("generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()


async def run_reporter(state: FactCheckState) -> str:
    """Run the reporter graph and return the markdown final report."""

    result = await build_reporter_graph().ainvoke(state)
    final_report = result.get("final_report")
    if not isinstance(final_report, str):
        raise ValueError("Reporter did not produce a markdown final_report string.")
    return final_report
