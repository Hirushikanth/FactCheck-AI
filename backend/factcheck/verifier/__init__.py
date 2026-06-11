"""Public entry point for claim verification."""

from __future__ import annotations

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.state import ClaimResult
from factcheck.verifier import graph as verifier_graph
from factcheck.verifier.schemas import VerifierState


async def run_verifier(claim: ValidatedClaim | str) -> ClaimResult:
    """Verify one extracted claim and return a structured result."""

    if isinstance(claim, ValidatedClaim):
        state = VerifierState(
            claim_text=claim.claim_text,
            source_sentence=claim.original_sentence,
            disambiguated_sentence=claim.disambiguated_sentence,
            original_index=claim.original_index,
            fidelity_status=claim.fidelity_status,
        )
        claim_text = claim.claim_text
    else:
        state = VerifierState(claim_text=claim)
        claim_text = claim

    result = await verifier_graph.build_verifier_graph().ainvoke(state)
    claim_result = result.get("claim_result")
    if claim_result is None:
        raise ValueError(f"Verifier did not produce a claim result for: {claim_text}")
    return claim_result
