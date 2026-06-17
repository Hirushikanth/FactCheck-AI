"""Verifier node for the main fact-checking pipeline."""

from __future__ import annotations

import asyncio
import logging

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.graph.event_bus import push_event
from factcheck.state import ClaimResult, FactCheckState
from factcheck.verifier import run_verifier
from factcheck.verifier.utils.claim_result import build_claim_result, is_processing_error


logger = logging.getLogger(__name__)


def _claim_text(claim: ValidatedClaim | str) -> str:
    return claim.claim_text if isinstance(claim, ValidatedClaim) else claim


def _source_sentence(claim: ValidatedClaim | str) -> str | None:
    return claim.original_sentence if isinstance(claim, ValidatedClaim) else None


def _fidelity_status(claim: ValidatedClaim | str) -> str | None:
    return claim.fidelity_status if isinstance(claim, ValidatedClaim) else None


def _make_error_result(claim: ValidatedClaim | str, exc: Exception) -> ClaimResult:
    return build_claim_result(
        claim=_claim_text(claim),
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        reasoning="Verification could not be completed due to a system error.",
        source_sentence=_source_sentence(claim),
        fidelity_status=_fidelity_status(claim),
        processing_status="error",
        processing_error=str(exc),
    )


async def _verify_single_claim(claim: ValidatedClaim | str) -> ClaimResult:
    """Verify one claim, returning an error result on failure."""
    try:
        return await run_verifier(claim)
    except Exception as exc:
        logger.error(
            "[verifier] Failed to verify claim '%s': %s",
            _claim_text(claim)[:80],
            exc,
        )
        return _make_error_result(claim, exc)


async def _verify_single_claim_with_event(
    session_id: str,
    claim: ValidatedClaim | str,
    index: int,
    total: int,
) -> ClaimResult:
    """Verify one claim and push an SSE event when done."""
    result = await _verify_single_claim(claim)
    await push_event(
        session_id,
        "verdict_ready",
        {
            "claim": result["claim"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "index": index,
            "total": total,
        },
    )
    return result


async def verifier_node(
    state: FactCheckState,
) -> dict[str, list[ClaimResult] | str]:
    """Verify all extracted claims in parallel and return all results at once."""
    claims = state.get("extracted_claims", [])
    session_id = state["session_id"]

    if not claims:
        logger.info("[verifier] No claims to verify.")
        return {
            "current_agent": "verifier",
            "claim_results": [],
        }

    logger.info("[verifier] Starting parallel verification of %d claims.", len(claims))

    results = await asyncio.gather(
        *(
            _verify_single_claim_with_event(session_id, claim, index, len(claims))
            for index, claim in enumerate(claims)
        ),
    )

    claim_results: list[ClaimResult] = list(results)

    error_count = sum(1 for result in claim_results if is_processing_error(result))
    if error_count > 0:
        logger.warning(
            "[verifier] %d/%d claims had verification errors.",
            error_count,
            len(claims),
        )

    logger.info("[verifier] Completed verification of %d claims.", len(claim_results))

    return {
        "current_agent": "verifier",
        "claim_results": claim_results,
    }
