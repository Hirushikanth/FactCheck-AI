"""Claim extraction package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from factcheck.extractor.schemas import ExtractorStageFailure, ExtractorState, ValidatedClaim
from factcheck.extractor.utils.assertion_profile import ResolvedExtractionMode


@dataclass(frozen=True)
class ExtractorRunResult:
    """Claims and stage failures from a full extractor run."""

    claims: list[ValidatedClaim]
    stage_failures: list[ExtractorStageFailure]
    resolved_extraction_mode: ResolvedExtractionMode | None = None
    selection_skipped: bool = False

    def __iter__(self):
        return iter(self.claims)

    def __len__(self) -> int:
        return len(self.claims)


async def run_extractor(
    raw_input: str,
    metadata: str | None = None,
    extraction_mode: Literal["auto", "claim", "document"] = "auto",
) -> ExtractorRunResult:
    """Run the claim extractor and return validated claims with stage failures."""

    from factcheck.extractor import graph as extractor_graph

    graph = extractor_graph.build_extractor_graph()
    result = await graph.ainvoke(
        ExtractorState(
            raw_input=raw_input,
            metadata=metadata,
            extraction_mode=extraction_mode,
        )
    )
    validated_claims = result.get("validated_claims", [])
    stage_failures_raw = result.get("stage_failures", [])
    resolved_mode = result.get("resolved_extraction_mode")

    claims: list[ValidatedClaim] = []
    for claim in validated_claims:
        if isinstance(claim, ValidatedClaim):
            claims.append(claim)
        elif isinstance(claim, dict):
            claims.append(ValidatedClaim.model_validate(claim))

    stage_failures: list[ExtractorStageFailure] = []
    for failure in stage_failures_raw:
        if isinstance(failure, ExtractorStageFailure):
            stage_failures.append(failure)
        elif isinstance(failure, dict):
            stage_failures.append(ExtractorStageFailure.model_validate(failure))

    return ExtractorRunResult(
        claims=claims,
        stage_failures=stage_failures,
        resolved_extraction_mode=resolved_mode,
        selection_skipped=resolved_mode == "direct_claim",
    )
