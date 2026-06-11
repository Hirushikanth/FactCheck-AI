"""Core reporter node for final fact-check report generation."""

from __future__ import annotations

import logging
from itertools import zip_longest
from typing import Any

from factcheck.config import get_settings
from factcheck.extractor.schemas import ValidatedClaim
from factcheck.llm.factory import get_reporter_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.reporter import config
from factcheck.reporter.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    build_summary_user_message,
    format_verdict_lines,
)
from factcheck.reporter.schemas import (
    FactCheckReport,
    ReportStatistics,
    ReportVerdict,
    ReportedClaim,
    SourceCitation,
    SummaryOutput,
)
from factcheck.state import ClaimResult, FactCheckState


logger = logging.getLogger(__name__)


def _coerce_verdict(raw_verdict: str) -> ReportVerdict:
    try:
        return ReportVerdict(raw_verdict)
    except ValueError:
        return ReportVerdict.INSUFFICIENT_EVIDENCE


def _calculate_statistics(claim_results: list[ClaimResult]) -> ReportStatistics:
    """Compute report statistics without using an LLM."""

    counts = {
        ReportVerdict.SUPPORTED: 0,
        ReportVerdict.REFUTED: 0,
        ReportVerdict.INSUFFICIENT_EVIDENCE: 0,
        ReportVerdict.CONFLICTING_EVIDENCE: 0,
    }
    for result in claim_results:
        counts[_coerce_verdict(result["verdict"])] += 1

    return ReportStatistics(
        total_claims=len(claim_results),
        supported=counts[ReportVerdict.SUPPORTED],
        refuted=counts[ReportVerdict.REFUTED],
        insufficient_evidence=counts[ReportVerdict.INSUFFICIENT_EVIDENCE],
        conflicting_evidence=counts[ReportVerdict.CONFLICTING_EVIDENCE],
    )


def _generate_template_summary(stats: ReportStatistics) -> str:
    """Generate a reliable fallback summary when the LLM cannot be used."""

    if stats.total_claims == 0:
        return (
            "No verifiable factual claims were identified in the submitted text. "
            "The content may be opinion-based, question-based, or too vague to verify "
            "against external evidence."
        )

    parts = [
        f"Fact-checking analysis identified {stats.total_claims} verifiable "
        f"claim{'s' if stats.total_claims != 1 else ''} in the submitted text."
    ]
    if stats.supported:
        parts.append(
            f"{stats.supported} claim{'s were' if stats.supported != 1 else ' was'} "
            "supported by available evidence."
        )
    if stats.refuted:
        parts.append(
            f"{stats.refuted} claim{'s were' if stats.refuted != 1 else ' was'} "
            "refuted by available evidence."
        )

    uncertain = stats.insufficient_evidence + stats.conflicting_evidence
    if uncertain:
        parts.append(
            f"{uncertain} claim{'s' if uncertain != 1 else ''} could not be "
            "verified conclusively because evidence was insufficient or conflicting."
        )

    parts.append(
        f"Overall credibility is {stats.credibility_label} "
        f"({stats.credibility_score:.0f}% of claims supported)."
    )
    return " ".join(parts)


async def _generate_llm_summary(
    *,
    original_text: str,
    stats: ReportStatistics,
    claim_results: list[ClaimResult],
) -> tuple[str, str]:
    """Generate a short executive summary through Ollama, with template fallback."""

    if stats.total_claims == 0:
        return _generate_template_summary(stats), "template"

    try:
        llm = get_reporter_llm(
            temperature=config.SUMMARY_TEMPERATURE,
            num_ctx=config.SUMMARY_NUM_CTX,
            num_predict=config.SUMMARY_MAX_PREDICT,
        )
        verdict_lines = format_verdict_lines(claim_results)
        user_message = build_summary_user_message(
            original_text=original_text,
            total_claims=stats.total_claims,
            supported=stats.supported,
            refuted=stats.refuted,
            insufficient_evidence=stats.insufficient_evidence,
            conflicting_evidence=stats.conflicting_evidence,
            credibility_score=stats.credibility_score,
            credibility_label=stats.credibility_label,
            verdict_lines=verdict_lines,
        )
        output = await call_llm_with_structured_output(
            llm=llm,
            output_class=SummaryOutput,
            messages=[
                ("system", SUMMARY_SYSTEM_PROMPT),
                ("human", user_message),
            ],
            context_desc="report summary",
        )
        if output is None or not output.summary.strip():
            raise ValueError("reporter summary output was empty")
        return output.summary.strip(), "llm"
    except Exception as exc:
        logger.warning("Reporter summary generation failed; using template: %s", exc)
        return _generate_template_summary(stats), "template"


def _claim_context(claim: ValidatedClaim | None, result: ClaimResult | None, index: int) -> dict[str, Any]:
    if claim is not None:
        return {
            "claim_text": claim.claim_text,
            "original_sentence": claim.original_sentence,
            "original_index": claim.original_index,
        }
    if result is not None:
        return {
            "claim_text": result["claim"],
            "original_sentence": result.get("source_sentence") or result["claim"],
            "original_index": index,
        }
    return {
        "claim_text": "",
        "original_sentence": "",
        "original_index": index,
    }


def _fallback_result_for_claim(claim: ValidatedClaim) -> ClaimResult:
    return {
        "claim": claim.claim_text,
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "evidence": [],
        "sources": [],
        "reasoning": "No verification result was produced for this extracted claim.",
        "search_queries": [],
        "source_sentence": claim.original_sentence,
        "fidelity_status": claim.fidelity_status,
    }


def _source_citations(result: ClaimResult) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    evidence = result.get("evidence", [])
    for index, url in enumerate(result.get("sources", [])):
        snippet = evidence[index] if index < len(evidence) else None
        citations.append(SourceCitation(url=url, snippet=snippet))
    return citations


def _build_reported_claims(
    extracted_claims: list[ValidatedClaim],
    claim_results: list[ClaimResult],
) -> list[ReportedClaim]:
    """Combine extractor context and verifier results into report-ready claims."""

    reported: list[ReportedClaim] = []
    for index, (claim, result) in enumerate(
        zip_longest(extracted_claims, claim_results, fillvalue=None)
    ):
        if claim is None and result is None:
            continue
        if result is None:
            result = _fallback_result_for_claim(claim)

        context = _claim_context(claim, result, index)
        reported.append(
            ReportedClaim(
                claim_text=context["claim_text"],
                original_sentence=context["original_sentence"],
                original_index=context["original_index"],
                verdict=_coerce_verdict(result["verdict"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                sources=_source_citations(result),
            )
        )

    return sorted(reported, key=lambda claim: (claim.original_index, claim.claim_text))


def _source_line(source: SourceCitation) -> str:
    line = f"- [{source.url}]({source.url})"
    if source.snippet:
        snippet = " ".join(source.snippet.split())
        if len(snippet) > config.MAX_SOURCE_SNIPPET_CHARS:
            snippet = f"{snippet[: config.MAX_SOURCE_SNIPPET_CHARS - 3].rstrip()}..."
        line += f" - {snippet}"
    return line


def render_markdown_report(report: FactCheckReport) -> str:
    """Render an internal report object into the shared markdown final_report field."""

    stats = report.statistics
    lines = [
        "# Fact-Check Report",
        "",
        "## Executive Summary",
        report.summary,
        "",
        "## Overview",
        f"- Total claims: {stats.total_claims}",
        (
            f"- Supported: {stats.supported} | Refuted: {stats.refuted} | "
            f"Insufficient evidence: {stats.insufficient_evidence} | "
            f"Conflicting evidence: {stats.conflicting_evidence}"
        ),
        f"- Overall credibility: {stats.credibility_label} ({stats.credibility_score:.0f}%)",
        "",
        "## Claims",
    ]

    if not report.claims:
        lines.extend(["", "No verifiable claims were available for detailed reporting."])
        return "\n".join(lines).strip() + "\n"

    previous_sentence: str | None = None
    for index, claim in enumerate(report.claims, start=1):
        if claim.original_sentence and claim.original_sentence != previous_sentence:
            lines.extend(["", f"### Source Sentence: {claim.original_sentence}"])
            previous_sentence = claim.original_sentence

        lines.extend(
            [
                "",
                f"### Claim {index} - {claim.verdict.value} (confidence: {claim.confidence:.2f})",
                f"**Statement:** {claim.claim_text}",
                f"**Explanation:** {claim.reasoning}",
                "**Sources:**",
            ]
        )
        if claim.sources:
            lines.extend(_source_line(source) for source in claim.sources)
        else:
            lines.append("- No sources recorded.")

    lines.extend(
        [
            "",
            "## Metadata",
            f"- Session ID: {report.session_id}",
            f"- Summary method: {report.generation_method}",
        ]
    )
    if report.model_used:
        lines.append(f"- Model: {report.model_used}")

    return "\n".join(lines).strip() + "\n"


async def generate_report_node(state: FactCheckState) -> dict[str, str]:
    """Generate the final markdown report. This node always returns a report string."""

    try:
        claim_results = list(state.get("claim_results", []))
        extracted_claims = list(state.get("extracted_claims", []))
        stats = _calculate_statistics(claim_results)
        summary, method = await _generate_llm_summary(
            original_text=state.get("raw_input", ""),
            stats=stats,
            claim_results=claim_results,
        )
        settings = get_settings()
        report = FactCheckReport(
            session_id=state.get("session_id", ""),
            original_text=state.get("raw_input", ""),
            summary=summary,
            statistics=stats,
            claims=_build_reported_claims(extracted_claims, claim_results),
            model_used=f"{settings.ollama_model} (Ollama)",
            generation_method=method,
        )
        final_report = render_markdown_report(report)
    except Exception as exc:
        logger.exception("Reporter failed unexpectedly; returning minimal fallback report.")
        stats = _calculate_statistics([])
        report = FactCheckReport(
            session_id=state.get("session_id", ""),
            original_text=state.get("raw_input", ""),
            summary=(
                "The reporter encountered an internal error while assembling the final "
                f"report: {exc}"
            ),
            statistics=stats,
            claims=[],
            generation_method="template",
        )
        final_report = render_markdown_report(report)

    return {
        "current_agent": "reporter",
        "final_report": final_report,
        "status": "done",
    }
