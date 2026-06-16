"""Extractor node for the main fact-checking pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from factcheck.extractor import run_extractor
from factcheck.extractor.schemas import ValidatedClaim
from factcheck.graph.event_bus import push_event
from factcheck.state import FactCheckState


def _unique_claims(claims: list[ValidatedClaim]) -> list[ValidatedClaim]:
    """Dedupe claims case-insensitively while preserving first-seen order."""
    unique: list[ValidatedClaim] = []
    seen: set[str] = set()
    for claim in claims:
        normalized_claim = claim.claim_text.strip()
        if not normalized_claim:
            continue

        key = normalized_claim.casefold()
        if key in seen:
            continue

        seen.add(key)
        unique.append(claim)

    return unique


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def extractor_node(state: FactCheckState) -> dict[str, list[ValidatedClaim] | str]:
    """Populate extracted claims from the raw user input."""

    session_id = state["session_id"]
    result = await run_extractor(state["raw_input"])

    for failure in result.stage_failures:
        await push_event(
            session_id,
            "extractor_stage_failed",
            {
                "stage": failure.stage,
                "sentence": failure.sentence,
                "reason": failure.reason,
                "successes": failure.successes,
                "attempts": failure.attempts,
                "timestamp": _now_iso(),
            },
        )

    return {
        "current_agent": "extractor",
        "extracted_claims": _unique_claims(result.claims),
    }
