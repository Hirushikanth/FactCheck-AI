"""Node: acknowledge_new_claim

Generates a friendly 2-3 sentence acknowledgement telling the user that
their new claim has been queued for fact-checking.

Also appends the exchange to dialogue_history so the conversation record
is complete.
"""

from __future__ import annotations

import logging
import time

from factcheck.dialogue.prompts import new_claim_acknowledgement_prompt
from factcheck.dialogue.schemas import DialogueState, DialogueTurn
from factcheck.dialogue.utils.tokens import estimate_tokens
from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.factory import get_dialogue_acknowledge_llm

logger = logging.getLogger(__name__)

_FALLBACK_ACK = (
    "Your new claim has been queued for fact-checking. "
    "Results will appear shortly. "
    "In the meantime, feel free to ask about the previous fact-check results."
)


async def acknowledge_new_claim_node(state: DialogueState) -> dict:
    """Generate and return a new-claim acknowledgement message."""
    claim_text: str = state.get("new_claim_text") or state["current_user_message"]
    prompt = new_claim_acknowledgement_prompt(claim_text)

    try:
        llm = get_dialogue_acknowledge_llm()
        async with get_ollama_semaphore():
            response = await llm.ainvoke(prompt)
        ack: str = response.content.strip() or _FALLBACK_ACK
    except Exception as exc:
        logger.error("[dialogue][acknowledge_new_claim] LLM call failed: %s", exc)
        ack = _FALLBACK_ACK

    # Record the exchange in history
    user_turn = DialogueTurn(
        role="user",
        content=state["current_user_message"],
        timestamp=time.time(),
        intent="new_claim",
        token_estimate=estimate_tokens(state["current_user_message"]),
    )
    assistant_turn = DialogueTurn(
        role="assistant",
        content=ack,
        timestamp=time.time(),
        intent=None,
        token_estimate=estimate_tokens(ack),
    )
    updated_history = list(state.get("dialogue_history", [])) + [user_turn, assistant_turn]

    logger.info("[dialogue][acknowledge_new_claim] Acknowledgement sent.")

    return {
        "dialogue_response": ack,
        "dialogue_history": updated_history,
        "classified_intent": "new_claim",
        "rewritten_query": None,
    }
