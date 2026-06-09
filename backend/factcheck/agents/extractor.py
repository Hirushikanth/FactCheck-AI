"""Extractor node for the main fact-checking pipeline."""

from __future__ import annotations

from factcheck.extractor import run_extractor
from factcheck.extractor.schemas import ValidatedClaim
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


async def extractor_node(state: FactCheckState) -> dict[str, list[ValidatedClaim] | str]:
    """Populate extracted claims from the raw user input."""

    claims = await run_extractor(state["raw_input"])
    return {"current_agent": "extractor", "extracted_claims": _unique_claims(claims)}
