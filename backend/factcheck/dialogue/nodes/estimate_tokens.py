"""Node: estimate_tokens

Counts prompt tokens using the same windowing strategy as assemble_context
and sets the ``needs_compression`` flag when the total approaches the ceiling.

No LLM call — pure Python arithmetic.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.config import (
    COMPRESSION_THRESHOLD_TOKENS,
    COMPRESSION_THRESHOLD_TURNS,
    MAX_USER_MESSAGE_TOKENS,
    SLIDING_WINDOW_MAX_TURNS,
    SLIDING_WINDOW_TOKEN_BUDGET,
    SYSTEM_PROMPT_TOKENS,
)
from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import (
    estimate_tokens,
    estimate_turn_tokens,
    get_windowed_history,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)

_PROMPT_OVERHEAD = 100


def _estimate_prompt_tokens(state: DialogueState) -> int:
    """Estimate tokens for the generation prompt (mirrors assemble_context)."""
    fc_context_tokens = estimate_tokens(state.get("_compressed_fc_context", ""))

    summary_tokens = estimate_tokens(
        (state.get("conversation_summary") or {}).get("text", "")
    )

    user_msg = state.get("current_user_message", "")
    if estimate_tokens(user_msg) > MAX_USER_MESSAGE_TOKENS:
        user_msg = truncate_to_tokens(user_msg, MAX_USER_MESSAGE_TOKENS)
    current_tokens = estimate_tokens(user_msg)

    windowed = get_windowed_history(
        state.get("dialogue_history", []),
        max_turns=SLIDING_WINDOW_MAX_TURNS,
        max_tokens=SLIDING_WINDOW_TOKEN_BUDGET,
    )
    history_tokens = sum(estimate_turn_tokens(t) for t in windowed)

    return (
        SYSTEM_PROMPT_TOKENS
        + fc_context_tokens
        + summary_tokens
        + history_tokens
        + current_tokens
        + _PROMPT_OVERHEAD
    )


async def estimate_tokens_node(state: DialogueState) -> dict:
    """Estimate total prompt tokens and set the compression flag."""
    total = _estimate_prompt_tokens(state)
    history_len = len(state.get("dialogue_history", []))
    turns_outside_window = max(history_len - SLIDING_WINDOW_MAX_TURNS, 0)
    turn_based_compression = turns_outside_window > COMPRESSION_THRESHOLD_TURNS
    needs_compression = total > COMPRESSION_THRESHOLD_TOKENS or turn_based_compression

    logger.debug(
        "[dialogue][estimate_tokens] total=%d threshold=%d compress=%s",
        total,
        COMPRESSION_THRESHOLD_TOKENS,
        needs_compression,
    )

    return {
        "estimated_context_tokens": total,
        "needs_compression": needs_compression,
    }
