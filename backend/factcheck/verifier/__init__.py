"""Public entry point for claim verification."""

from __future__ import annotations

from factcheck.state import ClaimResult
from factcheck.verifier import graph as verifier_graph
from factcheck.verifier.schemas import VerifierState


async def run_verifier(claim: str) -> ClaimResult:
    """Verify one extracted claim and return a structured result."""

    result = await verifier_graph.build_verifier_graph().ainvoke(VerifierState(claim=claim))
    claim_result = result.get("claim_result")
    if claim_result is None:
        raise ValueError(f"Verifier did not produce a claim result for: {claim}")
    return claim_result
