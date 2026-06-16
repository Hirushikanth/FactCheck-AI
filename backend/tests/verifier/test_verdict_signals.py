from __future__ import annotations

from factcheck.verifier.schemas import EvidenceItem
from factcheck.verifier.utils.verdict_signals import (
    claim_asserts_overload_or_harm,
    count_authoritative_contradictions,
    missing_aspects_only_reference_thresholds,
    reasoning_suggests_contradiction,
    snippet_contradicts_overload,
)


def test_snippet_contradicts_overload_detects_cdc_style_language() -> None:
    snippet = "Receiving multiple vaccines at once does not overload the immune system."
    assert snippet_contradicts_overload(snippet) is True


def test_claim_asserts_overload_for_vaccine_claim() -> None:
    claim = "Vaccines overload your immune system if you take more than two in a year."
    assert claim_asserts_overload_or_harm(claim) is True


def test_count_authoritative_contradictions_requires_two_high_tier_sources() -> None:
    evidence = [
        EvidenceItem(
            url="https://www.cdc.gov/example",
            snippet="Multiple vaccines do not overload the immune system.",
            credibility_tier="high",
            relevance_score=0.9,
        ),
        EvidenceItem(
            url="https://www.gavi.org/example",
            snippet="Safe to receive multiple vaccines at once without overwhelming immunity.",
            credibility_tier="high",
            relevance_score=0.8,
        ),
    ]
    claim = "Vaccines overload your immune system if you take more than two in a year."

    indices = count_authoritative_contradictions(
        claim_text=claim,
        evidence=evidence,
        influential_indices=[],
    )

    assert indices == [1, 2]


def test_missing_aspects_only_reference_thresholds() -> None:
    assert missing_aspects_only_reference_thresholds(
        ["exact threshold of more than two vaccines per year"]
    )
    assert not missing_aspects_only_reference_thresholds(
        ["general mechanism of immune overload"]
    )


def test_reasoning_suggests_contradiction() -> None:
    assert reasoning_suggests_contradiction(
        "The evidence contains conflicting information regarding immune overload."
    )
    assert not reasoning_suggests_contradiction("No relevant evidence was found.")
