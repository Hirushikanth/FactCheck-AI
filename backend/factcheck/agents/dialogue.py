"""Dialogue Agent — thin wrapper for the pipeline orchestration layer.

This module provides ``run_dialogue_agent()``, which maps the completed
``FactCheckState`` fields into the dialogue module's input format and
delegates to ``run_dialogue()``.

The dialogue agent is NOT part of the main fact-check pipeline graph.
It runs on demand when a follow-up message arrives for a completed session.
"""

from __future__ import annotations

import logging
from typing import Optional

from factcheck.dialogue import (
    ConversationSummary,
    DialogueOutput,
    DialogueTurn,
    run_dialogue,
)
from factcheck.state import FactCheckState

logger = logging.getLogger(__name__)


async def run_dialogue_agent(
    state: FactCheckState,
    *,
    user_message: str,
    dialogue_history: Optional[list[DialogueTurn]] = None,
    conversation_summary: Optional[ConversationSummary] = None,
    compressed_fc_context: Optional[str] = None,
) -> DialogueOutput:
    """Run one dialogue turn for a completed fact-check session.

    Args:
        state:                The completed FactCheckState from the pipeline.
        user_message:         The user's follow-up message.
        dialogue_history:     Prior dialogue turns (pass [] for first turn).
        conversation_summary: Rolling compressed summary from prior turns.
        compressed_fc_context: Cached fact-check context block from a prior turn.

    Returns:
        DialogueOutput containing the response and updated history.
    """
    return await run_dialogue(
        session_id=state["session_id"],
        user_message=user_message,
        raw_input=state.get("raw_input", ""),
        claim_results=[dict(cr) for cr in state.get("claim_results", [])],
        final_report=state.get("final_report"),
        dialogue_history=dialogue_history,
        conversation_summary=conversation_summary,
        compressed_fc_context=compressed_fc_context,
    )
