"""Dialogue Agent StateGraph.

Standalone LangGraph graph that handles one dialogue turn:
  init_context → estimate_tokens → [compress_history →] classify_intent
  → [forward_to_pipeline → acknowledge_new_claim]
     OR
  → rewrite_query → assemble_context → generate_response → update_history

Build once at import time and reuse the compiled singleton.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from factcheck.dialogue.nodes.acknowledge_new_claim import acknowledge_new_claim_node
from factcheck.dialogue.nodes.assemble_context import assemble_context_node
from factcheck.dialogue.nodes.classify_intent import classify_intent_node
from factcheck.dialogue.nodes.compress_history import compress_history_node
from factcheck.dialogue.nodes.estimate_tokens import estimate_tokens_node
from factcheck.dialogue.nodes.forward_to_pipeline import forward_to_pipeline_node
from factcheck.dialogue.nodes.generate_response import generate_response_node
from factcheck.dialogue.nodes.init_context import init_context_node
from factcheck.dialogue.nodes.rewrite_query import rewrite_query_node
from factcheck.dialogue.nodes.update_history import update_history_node
from factcheck.dialogue.schemas import DialogueState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_estimate(state: DialogueState) -> str:
    """After token estimation: compress if over threshold, else classify."""
    return "compress_history" if state.get("needs_compression") else "classify_intent"


def _route_after_intent(state: DialogueState) -> str:
    """After intent classification: forward new claims, otherwise rewrite."""
    intent = state.get("classified_intent", "clarification")
    if intent == "new_claim":
        return "forward_to_pipeline"
    return "rewrite_query"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_dialogue_graph() -> StateGraph:
    """Build and compile the Dialogue Agent StateGraph.

    Returns a compiled graph ready for ``ainvoke`` calls.
    """
    graph: StateGraph = StateGraph(DialogueState)

    # Register nodes
    graph.add_node("init_context",          init_context_node)
    graph.add_node("estimate_tokens",       estimate_tokens_node)
    graph.add_node("compress_history",      compress_history_node)
    graph.add_node("classify_intent",       classify_intent_node)
    graph.add_node("rewrite_query",         rewrite_query_node)
    graph.add_node("assemble_context",      assemble_context_node)
    graph.add_node("generate_response",     generate_response_node)
    graph.add_node("update_history",        update_history_node)
    graph.add_node("forward_to_pipeline",   forward_to_pipeline_node)
    graph.add_node("acknowledge_new_claim", acknowledge_new_claim_node)

    # Entry point
    graph.add_edge(START, "init_context")
    graph.add_edge("init_context", "estimate_tokens")

    # After estimation: compress or classify
    graph.add_conditional_edges(
        "estimate_tokens",
        _route_after_estimate,
        {
            "compress_history": "compress_history",
            "classify_intent":  "classify_intent",
        },
    )

    # After compression: always classify
    graph.add_edge("compress_history", "classify_intent")

    # After classification: new_claim → pipeline, else → rewrite
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_intent,
        {
            "forward_to_pipeline": "forward_to_pipeline",
            "rewrite_query":       "rewrite_query",
        },
    )

    # Main response path
    graph.add_edge("rewrite_query",     "assemble_context")
    graph.add_edge("assemble_context",  "generate_response")
    graph.add_edge("generate_response", "update_history")
    graph.add_edge("update_history",    END)

    # New-claim path
    graph.add_edge("forward_to_pipeline",   "acknowledge_new_claim")
    graph.add_edge("acknowledge_new_claim", END)

    return graph.compile()


# Singleton compiled graph — imported by run_dialogue() and the thin agent wrapper
dialogue_graph = build_dialogue_graph()
