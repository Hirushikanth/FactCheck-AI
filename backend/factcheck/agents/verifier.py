"""Verifier node for the main fact-checking pipeline."""

from __future__ import annotations

from factcheck.state import ClaimResult, FactCheckState
from factcheck.verifier import run_verifier


async def verifier_node(
    state: FactCheckState,
) -> dict[str, list[ClaimResult] | str]:
    """Append one evidence-grounded result for the next extracted claim."""

    completed = len(state["claim_results"])
    claims = state["extracted_claims"]
    if completed >= len(claims):
        return {"current_agent": "verifier"}

    claim = claims[completed]
    try:
        result = await run_verifier(claim)
    except Exception as exc:
        result: ClaimResult = {
            "claim": claim,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "evidence": [],
            "sources": [],
            "reasoning": f"Verifier failed while processing this claim: {exc}",
            "search_queries": [],
        }
        return {
            "current_agent": "verifier",
            "claim_results": state["claim_results"] + [result],
            "error": str(exc),
            "status": "error",
        }

    return {"current_agent": "verifier", "claim_results": state["claim_results"] + [result]}
