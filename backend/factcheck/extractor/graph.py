"""LangGraph wrapper for the claim extractor pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factcheck.extractor.nodes.decomposition import decomposition_node
from factcheck.extractor.nodes.disambiguation import disambiguation_node
from factcheck.extractor.nodes.fidelity import fidelity_node
from factcheck.extractor.nodes.selection import selection_node
from factcheck.extractor.nodes.sentence_splitter import sentence_splitter_node
from factcheck.extractor.nodes.validation import validation_node
from factcheck.extractor.schemas import ExtractorState


def build_extractor_graph():
    """Build the sequential extractor subgraph."""

    graph = StateGraph(ExtractorState)
    graph.add_node("sentence_splitter", sentence_splitter_node)
    graph.add_node("selection", selection_node)
    graph.add_node("disambiguation", disambiguation_node)
    graph.add_node("decomposition", decomposition_node)
    graph.add_node("fidelity", fidelity_node)
    graph.add_node("validation", validation_node)

    graph.add_edge(START, "sentence_splitter")
    graph.add_edge("sentence_splitter", "selection")
    graph.add_edge("selection", "disambiguation")
    graph.add_edge("disambiguation", "decomposition")
    graph.add_edge("decomposition", "fidelity")
    graph.add_edge("fidelity", "validation")
    graph.add_edge("validation", END)

    return graph.compile()
