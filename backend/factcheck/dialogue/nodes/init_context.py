"""Node: init_context

Compresses the fact-check results into a compact, reusable context block.
Runs once on the first dialogue turn; subsequent turns reuse the cached value.
No LLM call — pure Python transformation.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.prompts import compress_factcheck_context
from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


async def init_context_node(state: DialogueState) -> dict:
    """Build and cache the compressed fact-check context block.

    If ``_compressed_fc_context`` is already set (subsequent turns),
    this node is a no-op.
    """
    if state.get("_compressed_fc_context"):
        return {}  # already cached; skip

    claim_results: list[dict] = state.get("claim_results", [])
    fc_context = compress_factcheck_context(claim_results)

    token_count = estimate_tokens(fc_context)
    logger.debug(
        "[dialogue][init_context] Compressed %d claim(s) into %d tokens",
        len(claim_results),
        token_count,
    )

    return {"_compressed_fc_context": fc_context}
