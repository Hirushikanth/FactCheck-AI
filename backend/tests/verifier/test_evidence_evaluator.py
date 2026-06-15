from __future__ import annotations

from factcheck.verifier import prompts as verifier_prompts
from factcheck.verifier.nodes import evidence_evaluator
from factcheck.verifier.nodes.evidence_evaluator import (
    EvaluationOutput,
    _evaluation_messages,
    evidence_evaluator_node,
)
from factcheck.verifier.nodes.query_generator import _query_messages
from factcheck.verifier.schemas import CachedEvaluation, EvidenceItem, VerifierState


_BERRIES_SOURCE = (
    "Bananas are berries, but strawberries are not, "
    "according to the botanical definitions of fruits."
)
_BERRIES_CLAIM = (
    "Strawberries are not berries [according to botanical definitions of fruits]"
)


def test_verifier_prompts_require_definitional_framing() -> None:
    assert "botanical" in verifier_prompts.QUERY_GENERATOR_INITIAL_SYSTEM_PROMPT.casefold()
    assert "colloquial" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "CONFLICTING_EVIDENCE" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT
    assert "aggregate fruits" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "evidence excerpts" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "full-page article text" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "high-authority" in verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()


def test_format_evidence_labels_content_source() -> None:
    from factcheck.verifier.utils import format_evidence

    formatted = format_evidence(
        [
            EvidenceItem(
                url="https://science.example/earth",
                title="Earth shape",
                snippet="Fetched article text.",
                content_source="fetched",
                credibility_tier="high",
            ),
            EvidenceItem(
                url="https://news.example/earth",
                title="Earth news",
                snippet="Search snippet text.",
                content_source="snippet",
                credibility_tier="unknown",
            ),
        ]
    )

    assert "Excerpt (full-page excerpt): Fetched article text." in formatted
    assert "Excerpt (search snippet): Search snippet text." in formatted
    assert "Source tier: high-authority" in formatted
    assert "Source tier: unknown" in formatted


def test_evaluator_prompt_distinguishes_bracket_types() -> None:
    prompt = verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "definitional" in prompt or "domain framework" in prompt
    assert "geographic" in prompt or "temporal" in prompt or "jurisdictional" in prompt


def test_evaluator_prompt_requires_semantic_scope_matching() -> None:
    prompt = verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.casefold()
    assert "semantically" in prompt or "semantic" in prompt
    assert "literal" in prompt or "word overlap" in prompt


def test_evaluator_prompt_handles_non_definitional_brackets() -> None:
    prompt = verifier_prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT
    assert "in the United States" in prompt or "united states" in prompt.casefold()
    assert "do not apply definitional-frame rules" in prompt.casefold() or (
        "do not apply definitional" in prompt.casefold()
    )


def test_verifier_evaluator_prompt_includes_botanical_source_framing() -> None:
    state = VerifierState(
        claim_text=_BERRIES_CLAIM,
        source_sentence=_BERRIES_SOURCE,
        evidence=[
            EvidenceItem(
                url="https://example.com/berries",
                title="Berries",
                snippet="Strawberries are commonly called berries in everyday language.",
            )
        ],
    )

    human_prompt = _evaluation_messages(state)[-1][1]

    assert human_prompt.startswith("Evaluation frame:\n")
    assert "botanical definitions of fruits" in human_prompt
    assert _BERRIES_CLAIM in human_prompt


def test_verifier_query_prompt_includes_botanical_source_framing() -> None:
    state = VerifierState(
        claim_text=_BERRIES_CLAIM,
        source_sentence=_BERRIES_SOURCE,
    )

    human_prompt = _query_messages(state)[-1][1]

    assert "botanical definitions of fruits" in human_prompt
    assert _BERRIES_CLAIM in human_prompt


async def test_evidence_evaluator_passes_through_llm_verdict_without_guardrail(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="REFUTED",
            confidence=1.0,
            reasoning="Popular sources say strawberries are berries.",
            needs_more_evidence=False,
            missing_aspects=[],
            influential_sources=[1],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    evidence = [
        EvidenceItem(
            url="https://example.com/popular",
            snippet="Strawberries are commonly called berries in everyday language.",
        )
    ]

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text=_BERRIES_CLAIM,
            source_sentence=_BERRIES_SOURCE,
            all_queries=["strawberries botanical berries"],
            evidence=evidence,
        )
    )

    assert result["claim_result"]["verdict"] == "REFUTED"
    assert result["claim_result"]["confidence"] == 1.0
    assert result["claim_result"]["reasoning"] == "Popular sources say strawberries are berries."


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
        "source_sentence": "The Earth is an oblate spheroid.",
        "fidelity_status": None,
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


async def test_evidence_evaluator_finalizes_when_search_exhausted(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.3,
            reasoning="More evidence would help, but no new search query is available.",
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
            iteration_count=1,
            max_iterations=3,
            search_exhausted=True,
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["confidence"] == 0.3
    assert "iteration_count" not in result


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


async def test_evidence_evaluator_caches_successful_retry_assessment(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return EvaluationOutput(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.2,
            reasoning="The evidence does not address the exact figure.",
            needs_more_evidence=True,
            missing_aspects=["official source for the exact figure"],
            influential_sources=[1],
        )

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    evidence = [
        EvidenceItem(url="https://example.com", snippet="Related but vague."),
    ]

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A precise statistical claim.",
            evidence=evidence,
            iteration_count=1,
            max_iterations=3,
        )
    )

    assert result["cached_evaluation"] == CachedEvaluation(
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.2,
        reasoning="The evidence does not address the exact figure.",
        needs_more_evidence=True,
        missing_aspects=["official source for the exact figure"],
        influential_sources=[1],
    )


async def test_evidence_evaluator_falls_back_to_cached_evaluation(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return None

    monkeypatch.setattr(evidence_evaluator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(evidence_evaluator, "call_llm_with_structured_output", fake_structured_call)

    cached = CachedEvaluation(
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.2,
        reasoning="Three sources support partial context but lack the required document.",
        needs_more_evidence=True,
        missing_aspects=["official source for the exact figure"],
        influential_sources=[1, 2],
    )

    evidence = [
        EvidenceItem(url="https://first.example", snippet="First source."),
        EvidenceItem(url="https://second.example", snippet="Second source."),
    ]

    result = await evidence_evaluator_node(
        VerifierState(
            claim_text="A precise statistical claim.",
            evidence=evidence,
            iteration_count=2,
            max_iterations=3,
            cached_evaluation=cached,
        )
    )

    assert result["claim_result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert result["claim_result"]["confidence"] == 0.2
    assert (
        result["claim_result"]["reasoning"]
        == "Three sources support partial context but lack the required document."
    )
    assert result["intermediate_assessment"].needs_more_evidence is False
    assert result["intermediate_assessment"].missing_aspects == [
        "official source for the exact figure"
    ]
    assert evidence[0].is_influential is True
    assert evidence[1].is_influential is True
