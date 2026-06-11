"""Claim extraction package."""

from __future__ import annotations

from factcheck.extractor.schemas import ExtractorState, ValidatedClaim


async def run_extractor(raw_input: str, metadata: str | None = None) -> list[ValidatedClaim]:
    """Run the claim extractor and return validated claims with source context."""

    from factcheck.extractor import graph as extractor_graph

    graph = extractor_graph.build_extractor_graph()
    result = await graph.ainvoke(ExtractorState(raw_input=raw_input, metadata=metadata))
    validated_claims = result.get("validated_claims", [])

    claims: list[ValidatedClaim] = []
    for claim in validated_claims:
        if isinstance(claim, ValidatedClaim):
            claims.append(claim)
        elif isinstance(claim, dict):
            claims.append(ValidatedClaim.model_validate(claim))

    return claims
