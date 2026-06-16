"""Node: init_context

Compresses the fact-check results into a compact, reusable context block.
Runs once on the first dialogue turn; subsequent turns reuse the cached value
when it covers all completed runs.
No LLM call — pure Python transformation.
"""

from __future__ import annotations

import logging

from factcheck.dialogue.prompts import compress_factcheck_context, compress_factcheck_runs
from factcheck.dialogue.schemas import DialogueState
from factcheck.dialogue.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


async def init_context_node(state: DialogueState) -> dict:
    """Build and cache the compressed fact-check context block."""
    cached = state.get("_compressed_fc_context")
    covers = state.get("_fc_context_covers_sequence")
    latest = state.get("_latest_run_sequence", 0)

    if cached and covers is not None and covers >= latest:
        return {}

    runs = state.get("fact_check_runs") or []
    if runs:
        fc_context = compress_factcheck_runs(runs)
    else:
        claim_results: list[dict] = state.get("claim_results", [])
        fc_context = compress_factcheck_context(claim_results)

    token_count = estimate_tokens(fc_context)
    logger.debug(
        "[dialogue][init_context] Compressed %d run(s) into %d tokens",
        len(runs) if runs else len(state.get("claim_results", [])),
        token_count,
    )

    return {
        "_compressed_fc_context": fc_context,
        "_fc_context_covers_sequence": latest or (1 if runs else 0),
    }
