"""Claim extraction package."""

from __future__ import annotations

from factcheck.extractor.schemas import ExtractorState, ValidatedClaim


async def run_extractor(raw_input: str, metadata: str | None = None) -> list[str]:
    """Run the Claimify-style extractor and return validated claim text."""

    from factcheck.extractor import graph as extractor_graph

    graph = extractor_graph.build_extractor_graph()
    result = await graph.ainvoke(ExtractorState(raw_input=raw_input, metadata=metadata))
    validated_claims = result.get("validated_claims", [])

    claim_texts: list[str] = []
    for claim in validated_claims:
        if isinstance(claim, ValidatedClaim):
            claim_texts.append(claim.claim_text)
        elif isinstance(claim, dict) and isinstance(claim.get("claim_text"), str):
            claim_texts.append(claim["claim_text"])

    return claim_texts
