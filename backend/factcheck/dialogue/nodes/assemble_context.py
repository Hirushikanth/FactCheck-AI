"""Node: assemble_context

Pure Python — no LLM call.

Fits all prompt components within the token budget and builds the final
messages list that will be passed to the generation LLM.

Slot order (highest → lowest priority):
  1. System prompt + fact-check context   [never truncated]
  2. Conversation summary                  [dropped if no budget]
  3. Sliding window history                [trimmed from oldest turn first]
  4. Rewritten user message                [hard-truncated at 200 tokens]
  5. Intent hint + grounding reminder      [never truncated; small fixed size]
"""

from __future__ import annotations

import logging

from factcheck.dialogue.config import (
    MAX_RESPONSE_TOKENS,
    MAX_USER_MESSAGE_TOKENS,
    NUM_CTX,
    SLIDING_WINDOW_MAX_TURNS,
    SYSTEM_PROMPT_TOKENS,
)
from factcheck.dialogue.prompts import build_generator_messages, build_session_context_extras
from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import (
    estimate_tokens,
    get_windowed_history,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)

_PROMPT_OVERHEAD = 100  # encoding overhead budget (conservative)


async def assemble_context_node(state: DialogueState) -> dict:
    """Assemble the full prompt messages list within the token budget."""
    fc_context: str = state.get("_compressed_fc_context", "")
    extras = build_session_context_extras(
        state.get("original_text"),
        state.get("final_report"),
    )
    if extras:
        fc_context = f"{fc_context}\n\n{extras}" if fc_context else extras

    summary_text: str = (state.get("conversation_summary") or {}).get("text", "")

    user_msg: str = state.get("rewritten_query") or state["current_user_message"]

    # Hard-truncate user message at MAX_USER_MESSAGE_TOKENS
    user_msg_tokens = estimate_tokens(user_msg)
    if user_msg_tokens > MAX_USER_MESSAGE_TOKENS:
        user_msg = truncate_to_tokens(user_msg, MAX_USER_MESSAGE_TOKENS)
        logger.debug(
            "[dialogue][assemble_context] User message truncated from %d → %d tokens",
            user_msg_tokens,
            MAX_USER_MESSAGE_TOKENS,
        )

    # Calculate fixed token cost
    fixed_tokens = (
        SYSTEM_PROMPT_TOKENS
        + estimate_tokens(fc_context)
        + estimate_tokens(summary_text)
        + estimate_tokens(user_msg)
        + _PROMPT_OVERHEAD
    )

    # Budget remaining for history
    history_budget = max(NUM_CTX - MAX_RESPONSE_TOKENS - fixed_tokens, 0)

    windowed = get_windowed_history(
        state.get("dialogue_history", []),
        max_turns=SLIDING_WINDOW_MAX_TURNS,
        max_tokens=history_budget,
    )

    logger.debug(
        "[dialogue][assemble_context] fixed=%d history_budget=%d history_turns=%d",
        fixed_tokens,
        history_budget,
        len(windowed),
    )

    messages = build_generator_messages(
        fc_context_block=fc_context,
        summary=summary_text if summary_text else None,
        windowed_history=windowed,
        rewritten_query=user_msg,
        intent=state.get("classified_intent", "clarification"),
    )

    return {"_assembled_messages": messages}
