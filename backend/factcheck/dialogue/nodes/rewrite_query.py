"""Node: rewrite_query

Rewrites context-dependent follow-up messages into self-contained,
unambiguous queries that can be answered without reading prior history.

This matters especially for small 7B models, which have weaker coreference
resolution than larger models.  The rewrite step offloads coreference
resolution to a focused, single-purpose LLM call before the main generation.

Bypass conditions (skip the rewrite to save ~580 tokens):
  1. Intent is ``new_claim`` or ``out_of_scope`` — no rewriting needed.
  2. It is the first turn — no history to resolve references from.
  3. No coreference patterns detected via regex heuristic.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.prompts import build_rewriter_prompt, needs_rewriting
from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import get_windowed_history
from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.factory import get_dialogue_rewriter_llm

logger = logging.getLogger(__name__)

# Rewriter uses at most 4 turns and a tight token budget to stay lean
_REWRITER_MAX_TURNS = 4
_REWRITER_TOKEN_BUDGET = 600


async def rewrite_query_node(state: DialogueState) -> dict:
    """Rewrite the user message to be self-contained, or return it unchanged."""
    user_message: str = state["current_user_message"]
    intent: str = state.get("classified_intent", "clarification")

    # ── Bypass checks ──────────────────────────────────────────────────────
    if intent in ("new_claim", "out_of_scope"):
        logger.debug("[dialogue][rewrite_query] Bypass: intent=%s", intent)
        return {"rewritten_query": user_message}

    history = state.get("dialogue_history", [])
    if not history:
        logger.debug("[dialogue][rewrite_query] Bypass: first turn (no history)")
        return {"rewritten_query": user_message}

    if not needs_rewriting(user_message):
        logger.debug("[dialogue][rewrite_query] Bypass: no coreference patterns detected")
        return {"rewritten_query": user_message}

    # ── Rewrite ────────────────────────────────────────────────────────────
    windowed = get_windowed_history(
        history,
        max_turns=_REWRITER_MAX_TURNS,
        max_tokens=_REWRITER_TOKEN_BUDGET,
    )

    summary_text = (state.get("conversation_summary") or {}).get("text")
    prompt = build_rewriter_prompt(
        current_message=user_message,
        recent_history=windowed,
        summary=summary_text,
    )

    try:
        llm = get_dialogue_rewriter_llm()
        async with get_ollama_semaphore():
            response = await llm.ainvoke(prompt)

        rewritten: str = response.content.strip()

        # Sanity check: fall back to original if output is empty or suspiciously short
        if not rewritten or len(rewritten) < 5:
            rewritten = user_message

        logger.debug(
            "[dialogue][rewrite_query] '%s' → '%s'",
            user_message[:60],
            rewritten[:60],
        )
        return {"rewritten_query": rewritten}

    except Exception as exc:
        logger.error("[dialogue][rewrite_query] Rewrite failed: %s", exc)
        return {"rewritten_query": user_message}
