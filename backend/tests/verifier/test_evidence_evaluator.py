from __future__ import annotations

from factcheck.verifier.nodes import evidence_evaluator
from factcheck.verifier.nodes.evidence_evaluator import EvaluationOutput, evidence_evaluator_node
from factcheck.verifier.schemas import EvidenceItem, VerifierState


async def test_evidence_evaluator_maps_supported_output_to_claim_result(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        assert output_class is EvaluationOutput
        return EvaluationOutput(
            verdict="SUPPORTED",
            confidence=0.88,
            reasoning="Two sources directly support the claim.",
            needs_more_evidence=False,
            missing_aspects=[],
            influential_sources=[2],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    evidence = [
        EvidenceItem(url="https://first.example", title="First", snippet="Related evidence."),
        EvidenceItem(url="https://second.example", title="Second", snippet="Direct evidence."),
    ]

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            source_sentence="The Earth is an oblate spheroid.",
            all_queries=["earth oblate spheroid"],
            evidence=evidence,
        )
    )

    assert result["claim_result"] == {
        "claim": "The Earth is an oblate spheroid.",
        "verdict": "SUPPORTED",
        "confidence": 0.88,
        "evidence": ["Related evidence.", "Direct evidence."],
        "sources": ["https://first.example", "https://second.example"],
        "reasoning": "Two sources directly support the claim.",
        "search_queries": ["earth oblate spheroid"],
    }
    assert evidence[0].is_influential is False
    assert evidence[1].is_influential is True


async def test_evidence_evaluator_requests_retry_without_claim_result(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.2,
            reasoning="The evidence does not address the exact figure.",
            needs_more_evidence=True,
            missing_aspects=["official source for the exact figure"],
            influential_sources=[],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A precise statistical claim.",
            evidence=[EvidenceItem(url="https://example.com", snippet="Related but vague.")],
            iteration_count=1,
            max_iterations=3,
        )
    )

    assert "claim_result" not in result
    assert result["iteration_count"] == 2
    assert result["intermediate_assessment"].needs_more_evidence is True
    assert result["intermediate_assessment"].missing_aspects == [
        "official source for the exact figure"
    ]


async def test_evidence_evaluator_finalizes_when_retry_requested_at_max_iterations(
    monkeypatch,
) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.2,
            reasoning="More evidence would be useful, but the iteration cap has been reached.",
            needs_more_evidence=True,
            missing_aspects=["another source"],
            influential_sources=[],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A precise statistical claim.",
            evidence=[EvidenceItem(url="https://example.com", snippet="Related but vague.")],
            iteration_count=3,
            max_iterations=3,
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["confidence"] == 0.2


async def test_evidence_evaluator_finalizes_when_retry_would_exceed_max_iterations(
    monkeypatch,
) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.2,
            reasoning="More evidence would be useful, but no retry remains.",
            needs_more_evidence=True,
            missing_aspects=["another source"],
            influential_sources=[],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A precise statistical claim.",
            evidence=[EvidenceItem(url="https://example.com", snippet="Related but vague.")],
            iteration_count=2,
            max_iterations=3,
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["reasoning"] == "More evidence would be useful, but no retry remains."


async def test_evidence_evaluator_ignores_retry_request_for_decisive_verdicts(
    monkeypatch,
) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="SUPPORTED",
            confidence=0.8,
            reasoning="The evidence supports the claim.",
            needs_more_evidence=True,
            missing_aspects=["unnecessary extra source"],
            influential_sources=[1],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="The Earth orbits the Sun.",
            evidence=[EvidenceItem(url="https://example.com", snippet="Direct evidence.")],
            iteration_count=0,
            max_iterations=3,
        )
    )

    assert result["claim_result"]["verdict"] == "SUPPORTED"
    assert result["claim_result"]["confidence"] == 0.8


async def test_evidence_evaluator_falls_back_when_structured_output_fails(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return None

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A claim with unparsable evaluation.",
            evidence=[EvidenceItem(url="https://example.com", snippet="Evidence.")],
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["confidence"] == 0.0
    assert result["intermediate_assessment"].needs_more_evidence is False
