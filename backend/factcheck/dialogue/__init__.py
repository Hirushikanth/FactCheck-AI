"""Dialogue Agent — public entry point.

Usage::

    from factcheck.dialogue import run_dialogue, DialogueTurn, DialogueOutput

    result = await run_dialogue(
        session_id="abc-123",
        user_message="Why was Claim 2 rated REFUTED?",
        raw_input="The sky is green.",
        claim_results=[...],   # list[ClaimResult] dicts from FactCheckState
        final_report="...",
        dialogue_history=[],
        conversation_summary=None,
        compressed_fc_context=None,
    )
    print(result["response"])

The caller is responsible for persisting ``result["dialogue_history"]``,
``result["conversation_summary"]``, and ``result["compressed_fc_context"]``
between turns.  This module is stateless.
"""

from __future__ import annotations

import logging
from typing import Optional

from factcheck.dialogue.schemas import (
    ConversationSummary,
    DialogueOutput,
    DialogueState,
    DialogueTurn,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_dialogue",
    "DialogueTurn",
    "ConversationSummary",
    "DialogueOutput",
]


async def run_dialogue(
    *,
    session_id: str,
    user_message: str,
    raw_input: str,
    claim_results: list[dict],
    final_report: Optional[str] = None,
    dialogue_history: Optional[list[DialogueTurn]] = None,
    conversation_summary: Optional[ConversationSummary] = None,
    compressed_fc_context: Optional[str] = None,
    fact_check_runs: Optional[list[dict]] = None,
    latest_run_sequence: int = 0,
    fc_context_covers_sequence: Optional[int] = None,
) -> DialogueOutput:
    """Run one dialogue turn and return the result.

    Args:
        session_id:          Identifier for the current fact-check session.
        user_message:        The user's follow-up message for this turn.
        raw_input:           The original text that was fact-checked.
        claim_results:       List of ClaimResult dicts from the completed pipeline.
        final_report:        The reporter's markdown output (optional).
        dialogue_history:    Previous dialogue turns (pass [] for first turn).
        conversation_summary: Rolling summary from a prior compression run.
        compressed_fc_context: Cached fact-check context block from a prior turn.
        fact_check_runs:       Completed runs for cumulative context compression.
        latest_run_sequence:   Highest completed run sequence in the session.
        fc_context_covers_sequence: Run sequence covered by the cached context.

    Returns:
        DialogueOutput with the assistant's response, updated history, and
        pipeline handoff flags.
    """
    initial_state = DialogueState(
        session_id=session_id,
        original_text=raw_input,
        claim_results=claim_results,
        final_report=final_report,
        fact_check_runs=fact_check_runs or [],
        _latest_run_sequence=latest_run_sequence,
        _compressed_fc_context=compressed_fc_context,
        _fc_context_covers_sequence=fc_context_covers_sequence,
        dialogue_history=dialogue_history or [],
        conversation_summary=conversation_summary,
        current_user_message=user_message,
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    try:
        from factcheck.dialogue.graph import dialogue_graph

        result_state: DialogueState = await dialogue_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("[dialogue][run_dialogue] Graph error: %s", exc)
        return DialogueOutput(
            response="Sorry, I encountered an unexpected error. Please try again.",
            intent="clarification",
            dialogue_history=dialogue_history or [],
            conversation_summary=conversation_summary,
            compressed_fc_context=compressed_fc_context,
            fc_context_covers_sequence=fc_context_covers_sequence,
            needs_new_factcheck=False,
            new_claim_text=None,
            error=str(exc),
        )

    resolved_intent = result_state.get("classified_intent") or "clarification"

    return DialogueOutput(
        response=result_state.get("dialogue_response", ""),
        intent=resolved_intent,
        dialogue_history=result_state.get("dialogue_history", []),
        conversation_summary=result_state.get("conversation_summary"),
        compressed_fc_context=result_state.get("_compressed_fc_context"),
        fc_context_covers_sequence=result_state.get("_fc_context_covers_sequence"),
        needs_new_factcheck=result_state.get("needs_new_factcheck", False),
        new_claim_text=result_state.get("new_claim_text"),
        error=result_state.get("error_message"),
    )
