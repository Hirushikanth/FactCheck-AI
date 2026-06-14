"""Node: update_history

Appends the completed user + assistant turn pair to dialogue_history
and resets all per-turn working fields.

No LLM call — pure Python.
"""

from __future__ import annotations

import logging
import time

from factcheck.dialogue.schemas import DialogueState, DialogueTurn
from factcheck.dialogue.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


async def update_history_node(state: DialogueState) -> dict:
    """Append the new turn pair to dialogue_history and clear working fields."""
    user_content = state.get("rewritten_query") or state["current_user_message"]
    user_turn = DialogueTurn(
        role="user",
        content=user_content,
        timestamp=time.time(),
        intent=state.get("classified_intent"),
        token_estimate=estimate_tokens(user_content),
    )

    response_text = state.get("dialogue_response", "")
    assistant_turn = DialogueTurn(
        role="assistant",
        content=response_text,
        timestamp=time.time(),
        intent=None,
        token_estimate=estimate_tokens(response_text),
    )

    updated_history = list(state.get("dialogue_history", [])) + [user_turn, assistant_turn]

    logger.debug(
        "[dialogue][update_history] History now %d turns", len(updated_history)
    )

    return {
        "dialogue_history": updated_history,
        # Reset per-turn working fields; classified_intent is kept for output mapping
        "rewritten_query": None,
        "_assembled_messages": None,
    }
