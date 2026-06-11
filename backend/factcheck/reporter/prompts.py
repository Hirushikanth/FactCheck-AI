"""Prompt helpers for the reporter agent."""

from __future__ import annotations

from factcheck.reporter import config
from factcheck.state import ClaimResult


SUMMARY_SYSTEM_PROMPT = """\
You are a fact-checking report generator. Your task is to write a brief executive
summary of a completed fact-checking result.

Respond with ONLY a valid JSON object. Do not include markdown, code fences, or
extra explanation.

Required JSON structure:
{"summary": "your 2 to 4 sentence summary here"}

Rules:
1. Write 2 to 4 complete sentences.
2. Mention the total number of claims checked.
3. State how many claims were supported and refuted.
4. Include the overall credibility label.
5. Be neutral, concise, and evidence-grounded.
6. Do not add facts that are not present in the provided results.
"""


def _truncate(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return f"{clean[: max_chars - 15].rstrip()}... [truncated]"


def format_verdict_lines(
    claim_results: list[ClaimResult],
    *,
    max_claims: int = config.MAX_CLAIMS_IN_PROMPT,
) -> str:
    """Format claim verdicts for the summary prompt within a small token budget."""

    lines: list[str] = []
    for index, result in enumerate(claim_results[:max_claims], start=1):
        claim_text = _truncate(result["claim"], config.MAX_CLAIM_TEXT_CHARS)
        reasoning = _truncate(result["reasoning"], config.MAX_REASONING_CHARS)
        lines.append(f"{index}. [{result['verdict']}] {claim_text}\n   Reason: {reasoning}")

    remaining = len(claim_results) - max_claims
    if remaining > 0:
        lines.append(f"... and {remaining} more claim(s) not shown.")

    return "\n\n".join(lines) if lines else "No claim verdicts were produced."


def build_summary_user_message(
    *,
    original_text: str,
    total_claims: int,
    supported: int,
    refuted: int,
    insufficient_evidence: int,
    conflicting_evidence: int,
    credibility_score: float,
    credibility_label: str,
    verdict_lines: str,
) -> str:
    """Build the human-turn message for reporter summary generation."""

    return f"""Original text submitted for fact-checking:
{_truncate(original_text, config.MAX_INPUT_PREVIEW_CHARS)}

Fact-checking results:
- Total claims checked: {total_claims}
- Supported: {supported}
- Refuted: {refuted}
- Insufficient evidence: {insufficient_evidence}
- Conflicting evidence: {conflicting_evidence}
- Credibility score: {credibility_score:.0f}%
- Credibility label: {credibility_label}

Claims and verdicts:
{verdict_lines}

Write the executive summary JSON now."""
