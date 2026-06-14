"""Node: forward_to_pipeline

Marks the state so the caller knows a new fact-check is requested.
Sets ``needs_new_factcheck=True`` and extracts the raw claim text.

No LLM call — pure Python.  The caller (FastAPI route or wrapper) is
responsible for actually triggering the pipeline.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.schemas import DialogueState

logger = logging.getLogger(__name__)

# Common prefixes users add when submitting a new claim in the chat
_CLAIM_PREFIXES = [
    "can you also check:",
    "can you check this:",
    "fact-check this:",
    "fact check this:",
    "check this:",
    "also check:",
    "new claim:",
    "please check:",
    "verify this:",
]


async def forward_to_pipeline_node(state: DialogueState) -> dict:
    """Extract the new claim text and set the pipeline handoff flag."""
    raw_message: str = state["current_user_message"]
    claim_text = raw_message

    # Strip common prefixes to isolate the actual claim
    lower = raw_message.lower()
    for prefix in _CLAIM_PREFIXES:
        if lower.startswith(prefix):
            claim_text = raw_message[len(prefix):].strip()
            break

    logger.info(
        "[dialogue][forward_to_pipeline] New claim queued: '%s'", claim_text[:80]
    )

    return {
        "needs_new_factcheck": True,
        "new_claim_text": claim_text,
    }
