"""Node: generate_response

Main LLM call that produces the grounded dialogue response.
Uses the assembled messages from assemble_context_node.

One LLM call (~2500 tokens in / ~300 tokens out typical).
"""

from __future__ import annotations

import logging

from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import estimate_tokens
from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.factory import get_dialogue_llm

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE = (
    "I'm unable to generate a response right now. "
    "Please try rephrasing your question."
)

_CONTEXT_FAIL_RESPONSE = (
    "I encountered an error assembling the context for your question. "
    "Please try again."
)


async def generate_response_node(state: DialogueState) -> dict:
    """Generate the grounded dialogue response using the assembled context."""
    messages: list[dict] | None = state.get("_assembled_messages")

    if not messages:
        logger.error("[dialogue][generate_response] No assembled messages found.")
        return {
            "dialogue_response": _CONTEXT_FAIL_RESPONSE,
            "error_message": "context_assembly_failed",
        }

    try:
        llm = get_dialogue_llm()
        async with get_ollama_semaphore():
            response = await llm.ainvoke(messages)

        response_text: str = response.content.strip()

        if not response_text:
            response_text = _FALLBACK_RESPONSE

        logger.info(
            "[dialogue][generate_response] %d chars (~%d tokens)",
            len(response_text),
            estimate_tokens(response_text),
        )
        return {"dialogue_response": response_text, "error_message": None}

    except Exception as exc:
        logger.error("[dialogue][generate_response] Generation failed: %s", exc)
        return {
            "dialogue_response": (
                "Sorry, I encountered an error generating a response. Please try again."
            ),
            "error_message": str(exc),
        }
