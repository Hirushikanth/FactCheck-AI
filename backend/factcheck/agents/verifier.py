"""Phase 2 verifier placeholder."""

from __future__ import annotations

from factcheck.state import ClaimResult, FactCheckState


def verifier_node(state: FactCheckState) -> dict[str, list[ClaimResult] | str]:
    """Append one placeholder result until evidence verification is implemented."""

    completed = len(state["claim_results"])
    claims = state["extracted_claims"]
    if completed >= len(claims):
        return {"current_agent": "verifier"}

    claim = claims[completed]
    result: ClaimResult = {
        "claim": claim,
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "evidence": [],
        "sources": [],
        "reasoning": "Verifier evidence retrieval is implemented in a later phase.",
        "search_queries": [],
    }
    return {"current_agent": "verifier", "claim_results": state["claim_results"] + [result]}
