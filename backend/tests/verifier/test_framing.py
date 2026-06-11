from __future__ import annotations

from factcheck.verifier.schemas import EvidenceItem
from factcheck.verifier.utils.framing import (
    adjust_verdict_for_framing,
    colloquial_contradicts_framed_claim,
    extract_evaluation_frame,
    frame_aligned_refutes_framed_claim,
    frame_tokens,
    snippet_looks_colloquial,
    snippet_matches_frame,
)


_BERRIES_CLAIM = (
    "Strawberries are not berries [according to botanical definitions of fruits]"
)


def test_extract_evaluation_frame_from_bracketed_claim() -> None:
    assert extract_evaluation_frame(_BERRIES_CLAIM) == (
        "according to botanical definitions of fruits"
    )


def test_snippet_looks_colloquial_detects_popular_usage() -> None:
    assert snippet_looks_colloquial("Strawberries are commonly called berries.")
    assert not snippet_looks_colloquial(
        "Botanically, strawberries are aggregate fruits, not true berries."
    )


def test_snippet_matches_botanical_frame() -> None:
    tokens = frame_tokens("according to botanical definitions of fruits")

    assert snippet_matches_frame(
        "Botanically, strawberries are aggregate fruits, not true berries.",
        tokens,
    )
    assert not snippet_matches_frame("Strawberries are commonly called berries.", tokens)


def test_colloquial_contradicts_framed_negated_claim() -> None:
    assert colloquial_contradicts_framed_claim(
        _BERRIES_CLAIM,
        "Strawberries are commonly called berries in everyday language.",
    )


def test_adjust_verdict_downgrades_colloquial_only_refuted() -> None:
    verdict, confidence, reasoning = adjust_verdict_for_framing(
        claim_text=_BERRIES_CLAIM,
        verdict="REFUTED",
        confidence=1.0,
        reasoning="Popular sources say strawberries are berries.",
        evidence=[
            EvidenceItem(
                url="https://example.com/popular",
                snippet="Strawberries are commonly called berries in everyday language.",
            )
        ],
    )

    assert verdict == "CONFLICTING_EVIDENCE"
    assert confidence == 0.7
    assert "verdict adjusted" in reasoning.casefold()


def test_adjust_verdict_keeps_frame_aligned_refuted() -> None:
    frame_token_set = frame_tokens("according to botanical definitions of fruits")
    snippet = "Botanically, strawberries are classified as berries."
    assert frame_aligned_refutes_framed_claim(_BERRIES_CLAIM, snippet, frame_token_set)

    verdict, confidence, reasoning = adjust_verdict_for_framing(
        claim_text=_BERRIES_CLAIM,
        verdict="REFUTED",
        confidence=0.9,
        reasoning="Botanical source says strawberries are berries.",
        evidence=[EvidenceItem(url="https://example.com/botany", snippet=snippet)],
    )

    assert verdict == "REFUTED"
    assert confidence == 0.9
    assert reasoning == "Botanical source says strawberries are berries."


def test_adjust_verdict_leaves_unframed_claim_unchanged() -> None:
    verdict, confidence, reasoning = adjust_verdict_for_framing(
        claim_text="Strawberries are not berries.",
        verdict="REFUTED",
        confidence=1.0,
        reasoning="Evidence contradicts the claim.",
        evidence=[
            EvidenceItem(
                url="https://example.com/popular",
                snippet="Strawberries are commonly called berries.",
            )
        ],
    )

    assert verdict == "REFUTED"
    assert confidence == 1.0
    assert reasoning == "Evidence contradicts the claim."
