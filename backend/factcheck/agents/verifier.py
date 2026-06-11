"""Verifier node for the main fact-checking pipeline."""

from __future__ import annotations

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.state import ClaimResult, FactCheckState
from factcheck.verifier import run_verifier


def _claim_text(claim: ValidatedClaim | str) -> str:
    return claim.claim_text if isinstance(claim, ValidatedClaim) else claim


def _source_sentence(claim: ValidatedClaim | str) -> str | None:
    return claim.original_sentence if isinstance(claim, ValidatedClaim) else None


def _fidelity_status(claim: ValidatedClaim | str) -> str | None:
    return claim.fidelity_status if isinstance(claim, ValidatedClaim) else None


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
            "claim": _claim_text(claim),
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "evidence": [],
            "sources": [],
            "reasoning": f"Verifier failed while processing this claim: {exc}",
            "search_queries": [],
            "source_sentence": _source_sentence(claim),
            "fidelity_status": _fidelity_status(claim),
        }
        return {
            "current_agent": "verifier",
            "claim_results": state["claim_results"] + [result],
            "error": str(exc),
            "status": "error",
        }

    return {"current_agent": "verifier", "claim_results": state["claim_results"] + [result]}
