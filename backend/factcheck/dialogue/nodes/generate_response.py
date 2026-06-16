"""Node: generate_response

Main LLM call that produces the grounded dialogue response.
Uses the assembled messages from assemble_context_node.

One LLM call (~2500 tokens in / ~300 tokens out typical).
"""

from __future__ import annotations

import logging
from typing import Any

from factcheck.dialogue.config import DIALOGUE_NUM_PREDICT_RETRY
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


def _extract_response_text(response: Any) -> str:
    """Return stripped visible content from a LangChain AIMessage."""
    content = response.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _should_retry_empty_length(response: Any, text: str) -> bool:
    """True when the model hit num_predict with no visible answer text."""
    if text:
        return False
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("done_reason") == "length"


async def generate_response_node(state: DialogueState) -> dict:
    """Generate the grounded dialogue response using the assembled context."""
    messages: list[dict] | None = state.get("_assembled_messages")
    session_id = state.get("session_id", "unknown")

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

        response_text = _extract_response_text(response)

        if _should_retry_empty_length(response, response_text):
            meta = response.response_metadata or {}
            logger.warning(
                "[dialogue][generate_response] Empty content after length stop "
                "(session=%s eval_count=%s); retrying with num_predict=%d",
                session_id,
                meta.get("eval_count"),
                DIALOGUE_NUM_PREDICT_RETRY,
            )
            async with get_ollama_semaphore():
                response = await llm.bind(
                    num_predict=DIALOGUE_NUM_PREDICT_RETRY
                ).ainvoke(messages)
            response_text = _extract_response_text(response)

        if not response_text:
            meta = response.response_metadata or {}
            logger.warning(
                "[dialogue][generate_response] Using fallback (session=%s "
                "done_reason=%s eval_count=%s)",
                session_id,
                meta.get("done_reason"),
                meta.get("eval_count"),
            )
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
