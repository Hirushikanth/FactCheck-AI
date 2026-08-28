"""Tests for shared ClaimResult builder utilities."""

from __future__ import annotations

from factcheck.verifier.schemas import VerifierState
from factcheck.verifier.schemas import EvidenceItem
from factcheck.verifier.utils.claim_result import (
    build_claim_result,
    build_claim_result_from_state,
    is_processing_error,
)


def test_build_claim_result_omits_processing_fields_on_ok() -> None:
    result = build_claim_result(
        claim="The Earth is round.",
        verdict="SUPPORTED",
        confidence=0.9,
        reasoning="Supported by evidence.",
    )

    assert "processing_status" not in result
    assert "processing_error" not in result


def test_build_claim_result_sets_processing_fields_on_error() -> None:
    result = build_claim_result(
        claim="The Earth is round.",
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        reasoning="Verification could not be completed due to a system error.",
        processing_status="error",
        processing_error="boom",
    )

    assert result["processing_status"] == "error"
    assert result["processing_error"] == "boom"


def test_build_claim_result_sets_processing_fields_on_degraded() -> None:
    result = build_claim_result(
        claim="The Earth is round.",
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        reasoning="Fallback reasoning.",
        processing_status="degraded",
        processing_error="evaluator_fallback_no_verdict",
    )

    assert result["processing_status"] == "degraded"
    assert result["processing_error"] == "evaluator_fallback_no_verdict"


def test_build_claim_result_from_state_uses_verifier_state_fields() -> None:
    state = VerifierState(
        claim_text="Water boils at 100C.",
        source_sentence="Water boils at 100C at sea level.",
        fidelity_status="faithful",
        all_queries=["water boiling point"],
    )

    result = build_claim_result_from_state(
        state,
        verdict="SUPPORTED",
        confidence=0.8,
        reasoning="Supported.",
    )

    assert result["claim"] == "Water boils at 100C."
    assert result["search_queries"] == ["water boiling point"]
    assert result["source_sentence"] == "Water boils at 100C at sea level."
    assert result["fidelity_status"] == "faithful"
    assert "processing_status" not in result


def test_claim_result_limits_dialogue_evidence_to_two_short_snippets() -> None:
    state = VerifierState(
        claim_text="Water boils at 100C.",
        evidence=[
            EvidenceItem(
                url="https://example.com/one",
                snippet="one " * 100,
                is_influential=False,
            ),
            EvidenceItem(
                url="https://example.com/two",
                snippet="two " * 100,
                is_influential=True,
            ),
            EvidenceItem(
                url="https://example.com/three",
                snippet="three " * 100,
                is_influential=True,
            ),
        ],
    )

    result = build_claim_result_from_state(
        state,
        verdict="SUPPORTED",
        confidence=0.9,
        reasoning="Supported.",
    )

    assert len(result["dialogue_evidence"]) == 2
    assert result["dialogue_evidence"][0].startswith("two")
    assert all(len(item.split()) <= 60 for item in result["dialogue_evidence"])


def test_claim_result_uses_first_ranked_evidence_when_none_is_influential() -> None:
    state = VerifierState(
        claim_text="Water boils at 100C.",
        evidence=[
            EvidenceItem(url="https://example.com/one", snippet="first"),
            EvidenceItem(url="https://example.com/two", snippet="second"),
            EvidenceItem(url="https://example.com/three", snippet="third"),
        ],
    )

    result = build_claim_result_from_state(
        state,
        verdict="SUPPORTED",
        confidence=0.9,
        reasoning="Supported.",
    )

    assert result["dialogue_evidence"] == ["first", "second"]


def test_is_processing_error_only_true_for_error_status() -> None:
    assert is_processing_error({"processing_status": "error"}) is True
    assert is_processing_error({"processing_status": "degraded"}) is False
    assert is_processing_error({}) is False
