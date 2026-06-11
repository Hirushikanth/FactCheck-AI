from __future__ import annotations

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.state import ClaimResult, FactCheckState


def _validated_claim(
    claim_text: str,
    *,
    original_sentence: str | None = None,
    original_index: int = 0,
) -> ValidatedClaim:
    return ValidatedClaim(
        claim_text=claim_text,
        is_complete_declarative=True,
        disambiguated_sentence=claim_text,
        original_sentence=original_sentence or claim_text,
        original_index=original_index,
    )


def _claim_result(
    claim: str,
    verdict: str,
    *,
    confidence: float = 0.8,
    reasoning: str = "Evidence supports the claim.",
    evidence: list[str] | None = None,
    sources: list[str] | None = None,
) -> ClaimResult:
    return {
        "claim": claim,
        "verdict": verdict,  # type: ignore[typeddict-item]
        "confidence": confidence,
        "evidence": evidence or ["Evidence snippet."],
        "sources": sources or ["https://example.com/source"],
        "reasoning": reasoning,
        "search_queries": [f"{claim} evidence"],
    }


def _state(
    *,
    extracted_claims: list[ValidatedClaim] | None = None,
    claim_results: list[ClaimResult] | None = None,
) -> FactCheckState:
    return {
        "raw_input": "The Earth is round. The Moon is larger than the Sun.",
        "extracted_claims": extracted_claims or [],
        "claim_results": claim_results or [],
        "final_report": None,
        "messages": [],
        "current_agent": "",
        "session_id": "test-session",
        "error": None,
        "status": "running",
    }


def test_calculate_statistics_handles_all_verdicts() -> None:
    from factcheck.reporter.nodes.generate_report import _calculate_statistics

    stats = _calculate_statistics(
        [
            _claim_result("A", "SUPPORTED"),
            _claim_result("B", "REFUTED"),
            _claim_result("C", "INSUFFICIENT_EVIDENCE"),
            _claim_result("D", "CONFLICTING_EVIDENCE"),
        ]
    )

    assert stats.total_claims == 4
    assert stats.supported == 1
    assert stats.refuted == 1
    assert stats.insufficient_evidence == 1
    assert stats.conflicting_evidence == 1
    assert stats.credibility_score == 25.0
    assert stats.credibility_label == "Low"


def test_template_summary_handles_zero_claims() -> None:
    from factcheck.reporter.nodes.generate_report import (
        _calculate_statistics,
        _generate_template_summary,
    )

    summary = _generate_template_summary(_calculate_statistics([]))

    assert "No verifiable factual claims" in summary
    assert len(summary) > 20


def test_build_reported_claims_sorts_by_original_order_and_pairs_sources() -> None:
    from factcheck.reporter.nodes.generate_report import _build_reported_claims

    claims = [
        _validated_claim("Second claim.", original_sentence="Sentence 2.", original_index=2),
        _validated_claim("First claim.", original_sentence="Sentence 1.", original_index=0),
    ]
    results = [
        _claim_result(
            "Second claim.",
            "SUPPORTED",
            evidence=["Second evidence."],
            sources=["https://example.com/second"],
        ),
        _claim_result(
            "First claim.",
            "REFUTED",
            evidence=["First evidence."],
            sources=["https://example.com/first"],
        ),
    ]

    reported = _build_reported_claims(claims, results)

    assert [claim.claim_text for claim in reported] == ["First claim.", "Second claim."]
    assert reported[0].sources[0].url == "https://example.com/first"
    assert reported[0].sources[0].snippet == "First evidence."


def test_render_markdown_report_includes_verdict_confidence_reasoning_and_sources() -> None:
    from factcheck.reporter.nodes.generate_report import (
        _build_reported_claims,
        _calculate_statistics,
        render_markdown_report,
    )
    from factcheck.reporter.schemas import FactCheckReport

    claims = [_validated_claim("The Earth is round.", original_index=0)]
    results = [
        _claim_result(
            "The Earth is round.",
            "SUPPORTED",
            confidence=0.91,
            reasoning="Authoritative sources support this.",
            evidence=["NASA describes Earth as an oblate spheroid."],
            sources=["https://example.com/earth"],
        )
    ]
    reported = _build_reported_claims(claims, results)
    report = FactCheckReport(
        session_id="test-session",
        original_text="The Earth is round.",
        summary="One claim was checked and supported.",
        statistics=_calculate_statistics(results),
        claims=reported,
        generation_method="llm",
    )

    markdown = render_markdown_report(report)

    assert "# Fact-Check Report" in markdown
    assert "## Executive Summary" in markdown
    assert "### Claim 1 - SUPPORTED (confidence: 0.91)" in markdown
    assert "**Explanation:** Authoritative sources support this." in markdown
    assert "[https://example.com/earth](https://example.com/earth)" in markdown


async def test_generate_report_node_uses_llm_summary_when_available(monkeypatch) -> None:
    import factcheck.reporter.nodes.generate_report as reporter

    async def summary_stub(*args: object, **kwargs: object) -> tuple[str, str]:
        return "The submitted text contains one supported claim.", "llm"

    monkeypatch.setattr(reporter, "_generate_llm_summary", summary_stub)

    state = _state(
        extracted_claims=[_validated_claim("The Earth is round.")],
        claim_results=[_claim_result("The Earth is round.", "SUPPORTED")],
    )

    result = await reporter.generate_report_node(state)

    assert result["current_agent"] == "reporter"
    assert result["status"] == "done"
    assert "The submitted text contains one supported claim." in result["final_report"]
    assert "### Claim 1 - SUPPORTED" in result["final_report"]


async def test_generate_llm_summary_falls_back_when_structured_output_fails(monkeypatch) -> None:
    import factcheck.reporter.nodes.generate_report as reporter

    async def no_structured_output(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(reporter, "call_llm_with_structured_output", no_structured_output)
    monkeypatch.setattr(reporter, "get_reporter_llm", lambda **kwargs: object())

    stats = reporter._calculate_statistics([_claim_result("A", "SUPPORTED")])
    summary, method = await reporter._generate_llm_summary(
        original_text="A",
        stats=stats,
        claim_results=[_claim_result("A", "SUPPORTED")],
    )

    assert method == "template"
    assert "1 verifiable claim" in summary
