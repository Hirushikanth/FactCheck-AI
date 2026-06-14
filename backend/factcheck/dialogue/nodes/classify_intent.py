"""Node: classify_intent

Classifies the user message into one of five intent categories using a
single, low-token LLM call with fully deterministic settings.

Intent taxonomy:
  clarification     — asking about current session verdicts / evidence / sources
  general_question  — factual question about the topic (not the check results)
  new_claim         — submitting new text to be fact-checked
  out_of_scope      — completely off-topic
  ask_clarification — too vague; agent needs to ask a clarifying question

Single-word output is used (no JSON) because instruction-tuned 7B models
reliably produce one word when the prompt ends with ``LABEL:``.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.prompts import build_intent_prompt, parse_intent
from factcheck.dialogue.schemas import DialogueState
from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.factory import get_dialogue_classifier_llm

logger = logging.getLogger(__name__)

# Common stopwords to strip when extracting topic keywords from claims
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "that", "this", "it",
    "has", "have", "by", "from", "with", "be", "been", "its",
})


def _extract_topic_keywords(claim_results: list[dict], max_words: int = 12) -> str:
    """Quick keyword extraction from the first 3 claims for classifier context."""
    claims_text = " ".join(
        cr.get("claim", "") for cr in claim_results[:3]
    )
    all_words = claims_text.split()
    keywords = [
        w.strip(".,;:!?\"'").lower()
        for w in all_words
        if w.lower().strip(".,;:!?\"'") not in _STOPWORDS and len(w) > 3
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_keywords = [k for k in keywords if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]
    return ", ".join(unique_keywords[:max_words]) or "general topic"


async def classify_intent_node(state: DialogueState) -> dict:
    """Classify the user message and store the intent label in state."""
    claim_results = state.get("claim_results", [])
    topic_keywords = _extract_topic_keywords(claim_results)

    prompt = build_intent_prompt(
        user_message=state["current_user_message"],
        num_claims=len(claim_results),
        topic_keywords=topic_keywords,
    )

    try:
        llm = get_dialogue_classifier_llm()
        async with get_ollama_semaphore():
            response = await llm.ainvoke(prompt)

        intent = parse_intent(response.content)
        logger.info("[dialogue][classify_intent] → '%s'", intent)
        return {"classified_intent": intent}

    except Exception as exc:
        logger.error("[dialogue][classify_intent] Classification failed: %s", exc)
        return {"classified_intent": "clarification"}  # safe fallback
