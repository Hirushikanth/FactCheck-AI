"""Tests for dialogue prompt helpers.

Covers:
  - compress_factcheck_context() token budget
  - parse_intent() all labels + unknown fallback
  - needs_rewriting() regex patterns
"""

from __future__ import annotations

import pytest

from factcheck.dialogue.prompts import (
    compress_factcheck_context,
    needs_rewriting,
    parse_intent,
)
from factcheck.dialogue.utils.tokens import estimate_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_claim_result(
    claim: str = "The Earth is round.",
    verdict: str = "SUPPORTED",
    confidence: float = 0.91,
    evidence: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict:
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence or ["Scientists confirm spherical Earth."],
        "sources": sources or ["https://example.com/earth"],
        "reasoning": "Multiple independent sources support this.",
        "search_queries": ["Earth shape evidence"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# compress_factcheck_context
# ─────────────────────────────────────────────────────────────────────────────

def test_compress_factcheck_context_empty_results() -> None:
    """Empty claim list produces a valid 'no claims' block."""
    result = compress_factcheck_context([])
    assert "=== FACT-CHECK RESULTS" in result
    assert "No claims were checked" in result
    assert "=== END OF FACT-CHECK CONTEXT ===" in result


def test_compress_factcheck_context_single_claim() -> None:
    """Single claim block contains all key fields."""
    cr = _make_claim_result(verdict="REFUTED", confidence=0.85)
    result = compress_factcheck_context([cr])

    assert "[Claim 1]" in result
    assert "REFUTED" in result
    assert "85%" in result
    assert "example.com" in result


def test_compress_factcheck_context_ten_claims_under_800_tokens() -> None:
    """Ten claims must compress into ≤ 800 tokens."""
    claims = [
        _make_claim_result(
            claim=f"Claim number {i} with some descriptive text that fills space.",
            verdict="SUPPORTED" if i % 2 == 0 else "REFUTED",
            confidence=0.7 + (i * 0.02),
            evidence=[f"Evidence snippet {i} providing context about the claim."],
            sources=[f"https://source{i}.example.com/article"],
        )
        for i in range(10)
    ]
    result = compress_factcheck_context(claims)
    token_count = estimate_tokens(result)
    assert token_count <= 800, f"Context block is {token_count} tokens — exceeds 800"


def test_compress_factcheck_context_truncates_long_claim() -> None:
    """Claims longer than 120 chars are truncated with '...'."""
    long_claim = "A" * 200
    cr = _make_claim_result(claim=long_claim)
    result = compress_factcheck_context([cr])
    assert "..." in result
    # The full 200-char claim should not appear
    assert long_claim not in result


def test_compress_factcheck_context_all_verdict_types() -> None:
    """All four verdict types are rendered correctly."""
    verdicts = ["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]
    claims = [_make_claim_result(claim=f"Claim {v}.", verdict=v) for v in verdicts]
    result = compress_factcheck_context(claims)
    for verdict in verdicts:
        assert verdict in result


def test_compress_factcheck_context_no_sources() -> None:
    """Missing sources field produces 'No sources listed' rather than crashing."""
    cr = _make_claim_result()
    cr["sources"] = []
    result = compress_factcheck_context([cr])
    assert "No sources listed" in result


# ─────────────────────────────────────────────────────────────────────────────
# parse_intent
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("clarification",     "clarification"),
    ("general_question",  "general_question"),
    ("new_claim",         "new_claim"),
    ("out_of_scope",      "out_of_scope"),
    ("ask_clarification", "ask_clarification"),
    # With trailing punctuation
    ("clarification.",    "clarification"),
    ("new_claim,",        "new_claim"),
    # Unknown label → fallback
    ("hallucinated_label", "clarification"),
    ("",                  "clarification"),
    ("   ",               "clarification"),
    # Extra words after the label
    ("clarification this is some extra text", "clarification"),
    # Mixed case
    ("CLARIFICATION",     "clarification"),
    # Preamble before label
    ("The label is: new_claim", "new_claim"),
    ("LABEL: general_question", "general_question"),
])
def test_parse_intent(raw: str, expected: str) -> None:
    assert parse_intent(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# needs_rewriting
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("message,expected", [
    ("What did it say?",                    True),   # 'it'
    ("Can you explain this?",               True),   # 'this'
    ("What sources were used for that?",    True),   # 'that'
    ("The verdict for those claims?",       True),   # 'those'
    ("What is the verdict for the claim?",  True),   # 'the claim'
    ("What is the result?",                 True),   # 'the result'
    # Clear, standalone queries — no rewriting needed
    ("Why was Claim 2 rated REFUTED?",      False),
    ("What sources were used for Claim 1?", False),
    ("How many claims were checked?",       False),
])
def test_needs_rewriting(message: str, expected: bool) -> None:
    assert needs_rewriting(message) == expected
