"""Node: compress_history

Summarises conversation turns that have fallen outside the sliding window
into a rolling summary string.  Called only when ``needs_compression`` is True.

One LLM call per trigger (~400 tokens in, ~80 tokens out).
"""

from __future__ import annotations

import logging
import time

from factcheck.dialogue.config import MAX_SUMMARY_TOKENS, SLIDING_WINDOW_MAX_TURNS
from factcheck.dialogue.prompts import build_compressor_prompt
from factcheck.dialogue.schemas import ConversationSummary, DialogueState
from factcheck.dialogue.utils.tokens import estimate_tokens
from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.factory import get_dialogue_compressor_llm

logger = logging.getLogger(__name__)


async def compress_history_node(state: DialogueState) -> dict:
    """Summarise turns outside the sliding window into a rolling summary.

    The new summary replaces the old one (rather than chaining).  It is hard-
    capped at MAX_SUMMARY_TOKENS by truncating at the last sentence boundary.
    """
    history = state.get("dialogue_history", [])
    window_size = SLIDING_WINDOW_MAX_TURNS

    if len(history) <= window_size:
        # Nothing outside the window — clear the flag and move on.
        return {"needs_compression": False}

    turns_to_compress = history[: len(history) - window_size]
    existing_summary = (state.get("conversation_summary") or {}).get("text")

    prompt = build_compressor_prompt(turns_to_compress, existing_summary)

    try:
        llm = get_dialogue_compressor_llm()
        async with get_ollama_semaphore():
            response = await llm.ainvoke(prompt)

        summary_text: str = response.content.strip()

        # Hard cap at MAX_SUMMARY_TOKENS
        if estimate_tokens(summary_text) > MAX_SUMMARY_TOKENS:
            sentences = summary_text.split(". ")
            summary_text = ". ".join(sentences[:2])
            if not summary_text.endswith("."):
                summary_text += "."

        new_summary = ConversationSummary(
            text=summary_text,
            turns_compressed=len(turns_to_compress),
            last_updated=time.time(),
        )

        logger.info(
            "[dialogue][compress_history] Compressed %d turns → %d tokens",
            len(turns_to_compress),
            estimate_tokens(summary_text),
        )

        return {
            "conversation_summary": new_summary,
            "needs_compression": False,
        }

    except Exception as exc:
        logger.error("[dialogue][compress_history] Compression failed: %s", exc)
        # Non-fatal: continue without surfacing an error to the caller
        return {"needs_compression": False}
