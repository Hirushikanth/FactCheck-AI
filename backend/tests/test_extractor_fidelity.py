from __future__ import annotations

from factcheck.extractor import prompts
from factcheck.extractor.nodes import fidelity, validation
from factcheck.extractor.nodes.fidelity import FidelityAuditOutput, fidelity_node
from factcheck.extractor.nodes.validation import ValidationOutput, validation_node
from factcheck.extractor.schemas import ExtractorState, PotentialClaim
from factcheck.extractor.utils.fidelity import (
    CoverageDecision,
    FidelityDecision,
    _drops_scoped_negation,
    assess_claim_fidelity,
    assess_group_coverage,
)
from factcheck.verifier.nodes.evidence_evaluator import _evaluation_messages
from factcheck.verifier.nodes.query_generator import _query_messages
from factcheck.verifier.schemas import EvidenceItem, VerifierState


def _potential_claim(
    claim_text: str,
    *,
    disambiguated_sentence: str,
    original_sentence: str | None = None,
    original_index: int = 0,
) -> PotentialClaim:
    return PotentialClaim(
        claim_text=claim_text,
        disambiguated_sentence=disambiguated_sentence,
        original_sentence=original_sentence or disambiguated_sentence,
        original_index=original_index,
    )


def test_decomposition_prompt_requires_compound_but_splits_and_framing() -> None:
    prompt = prompts.DECOMPOSITION_SYSTEM_PROMPT

    assert "but" in prompt.casefold()
    assert "Bananas are berries [according to botanical definitions of fruits]" in prompt
    assert "Strawberries are not berries [according to botanical definitions of fruits]" in prompt
    assert "temporal or causal subordinate clause" in prompt.casefold()
    assert "The French Revolution began in 1815 after Napoleon's defeat." in prompt
    assert "Incorrect extraction: [\"Napoleon was defeated\"]" in prompt


def test_extractor_prompts_explicitly_forbid_truth_correction() -> None:
    for prompt in (
        prompts.SELECTION_SYSTEM_PROMPT,
        prompts.DISAMBIGUATION_SYSTEM_PROMPT,
        prompts.DECOMPOSITION_SYSTEM_PROMPT,
    ):
        assert "do not correct" in prompt.casefold()
        assert "The pyramids were built by aliens" in prompt
        assert "Drinking bleach cures COVID-19" in prompt


def test_programmatic_fidelity_rejects_truth_biased_entity_substitution() -> None:
    result = assess_claim_fidelity(
        claim_text="The pyramids were built by ancient Egyptians.",
        source_sentence="The pyramids were built by aliens.",
    )

    assert result.decision == FidelityDecision.FAIL
    assert {"ancient", "egyptians"}.issubset(result.extra_terms)


def test_programmatic_fidelity_preserves_dangerous_false_claims() -> None:
    result = assess_claim_fidelity(
        claim_text="Drinking bleach cures COVID-19.",
        source_sentence="Drinking bleach cures COVID-19.",
    )

    assert result.decision == FidelityDecision.PASS


def test_programmatic_fidelity_allows_tense_normalization() -> None:
    result = assess_claim_fidelity(
        claim_text="Jane runs the firm.",
        source_sentence="Jane was running the firm.",
    )

    assert result.decision == FidelityDecision.PASS


def test_programmatic_fidelity_allows_plural_to_singular_normalization() -> None:
    result = assess_claim_fidelity(
        claim_text="The pyramid was built by aliens.",
        source_sentence="The pyramids were built by aliens.",
    )

    assert result.decision == FidelityDecision.PASS


_BERRIES_SOURCE = (
    "Bananas are berries, but strawberries are not, "
    "according to the botanical definitions of fruits."
)

_NAPOLEON_SOURCE = "The French Revolution began in 1815 after Napoleon's defeat."

_ADA_BABBAGE_SOURCE = (
    "Ada Lovelace wrote notes about the Analytical Engine "
    "and Charles Babbage designed it."
)

_BURIED_CLAIM_SOURCE = (
    "John's notable research on neural networks demonstrates the power of innovation."
)


def test_programmatic_fidelity_allows_positive_conjunct_in_compound_sentence() -> None:
    result = assess_claim_fidelity(
        claim_text="Bananas are berries.",
        source_sentence=_BERRIES_SOURCE,
    )

    assert result.decision == FidelityDecision.PASS


def test_programmatic_fidelity_allows_negated_conjunct_in_compound_sentence() -> None:
    result = assess_claim_fidelity(
        claim_text="Strawberries are not berries.",
        source_sentence=_BERRIES_SOURCE,
    )

    assert result.decision == FidelityDecision.PASS


def test_programmatic_fidelity_rejects_negation_drop_on_same_entity() -> None:
    result = assess_claim_fidelity(
        claim_text="Strawberries are berries.",
        source_sentence="Strawberries are not berries.",
    )

    assert result.decision == FidelityDecision.FAIL
    assert "negation" in result.reason.casefold()


_MULTI_NEGATION_SOURCE = (
    "Strawberries are not berries and blueberries are not berries."
)


def test_programmatic_fidelity_rejects_partial_multi_negation_drop() -> None:
    result = assess_claim_fidelity(
        claim_text="Strawberries are not berries and blueberries are berries.",
        source_sentence=_MULTI_NEGATION_SOURCE,
    )

    assert result.decision == FidelityDecision.FAIL
    assert "negation" in result.reason.casefold()


def test_programmatic_fidelity_allows_full_multi_negation_preserved() -> None:
    result = assess_claim_fidelity(
        claim_text="Strawberries are not berries and blueberries are not berries.",
        source_sentence=_MULTI_NEGATION_SOURCE,
    )

    assert result.decision == FidelityDecision.PASS


def test_programmatic_fidelity_allows_partial_multi_negation_conjunct() -> None:
    result = assess_claim_fidelity(
        claim_text="Strawberries are not berries.",
        source_sentence=_MULTI_NEGATION_SOURCE,
    )

    assert result.decision == FidelityDecision.PASS


def test_drops_scoped_negation_allows_negated_subject_in_multi_source() -> None:
    dropped = _drops_scoped_negation(
        "Strawberries are not berries.",
        "Neither strawberries nor blueberries are berries.",
    )

    assert dropped == set()


async def test_fidelity_node_preserves_both_compound_conjuncts(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM audit should not run for faithful compound splits")

    monkeypatch.setattr(fidelity, "_audit_claim_fidelity", fail_if_called)
    state = ExtractorState(
        raw_input=_BERRIES_SOURCE,
        potential_claims=[
            _potential_claim(
                "Bananas are berries [according to botanical definitions of fruits]",
                disambiguated_sentence=_BERRIES_SOURCE,
            ),
            _potential_claim(
                "Strawberries are not berries [according to botanical definitions of fruits]",
                disambiguated_sentence=_BERRIES_SOURCE,
            ),
        ],
    )

    result = await fidelity_node(state)

    assert [claim.claim_text for claim in result["potential_claims"]] == [
        "Bananas are berries [according to botanical definitions of fruits]",
        "Strawberries are not berries [according to botanical definitions of fruits]",
    ]


def test_programmatic_fidelity_allows_legitimate_atomic_splits() -> None:
    source = "Ada Lovelace wrote notes about the Analytical Engine and Charles Babbage designed it."

    first = assess_claim_fidelity(
        claim_text="Ada Lovelace wrote notes about the Analytical Engine.",
        source_sentence=source,
    )
    second = assess_claim_fidelity(
        claim_text="Charles Babbage designed the Analytical Engine.",
        source_sentence=source,
    )

    assert first.decision == FidelityDecision.PASS
    assert second.decision == FidelityDecision.PASS


def test_group_coverage_marks_napoleon_subset_incomplete() -> None:
    result = assess_group_coverage(
        source_sentence=_NAPOLEON_SOURCE,
        claim_texts=["Napoleon was defeated"],
    )

    assert result.decision == CoverageDecision.INCOMPLETE
    assert result.uncovered_segments


def test_group_coverage_accepts_full_napoleon_split() -> None:
    result = assess_group_coverage(
        source_sentence=_NAPOLEON_SOURCE,
        claim_texts=[
            "The French Revolution began in 1815",
            "Napoleon was defeated",
        ],
    )

    assert result.decision == CoverageDecision.COMPLETE


def test_group_coverage_accepts_berries_compound() -> None:
    result = assess_group_coverage(
        source_sentence=_BERRIES_SOURCE,
        claim_texts=[
            "Bananas are berries [according to botanical definitions of fruits]",
            "Strawberries are not berries [according to botanical definitions of fruits]",
        ],
    )

    assert result.decision == CoverageDecision.COMPLETE


def test_group_coverage_accepts_ada_babbage_split() -> None:
    result = assess_group_coverage(
        source_sentence=_ADA_BABBAGE_SOURCE,
        claim_texts=[
            "Ada Lovelace wrote notes about the Analytical Engine.",
            "Charles Babbage designed the Analytical Engine.",
        ],
    )

    assert result.decision == CoverageDecision.COMPLETE


def test_group_coverage_marks_ada_babbage_single_claim_incomplete() -> None:
    result = assess_group_coverage(
        source_sentence=_ADA_BABBAGE_SOURCE,
        claim_texts=["Ada Lovelace wrote notes about the Analytical Engine."],
    )

    assert result.decision == CoverageDecision.INCOMPLETE


def test_group_coverage_leaves_simple_false_claim_complete() -> None:
    result = assess_group_coverage(
        source_sentence="Drinking bleach cures COVID-19.",
        claim_texts=["Drinking bleach cures COVID-19."],
    )

    assert result.decision == CoverageDecision.COMPLETE


def test_group_coverage_leaves_buried_claim_partial_extraction_complete() -> None:
    result = assess_group_coverage(
        source_sentence=_BURIED_CLAIM_SOURCE,
        claim_texts=["John has research on neural networks."],
    )

    assert result.decision == CoverageDecision.COMPLETE


async def test_fidelity_node_falls_back_on_incomplete_group_coverage(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM audit should not run for faithful incomplete decomposition")

    monkeypatch.setattr(fidelity, "_audit_claim_fidelity", fail_if_called)
    state = ExtractorState(
        raw_input=_NAPOLEON_SOURCE,
        potential_claims=[
            _potential_claim(
                "Napoleon was defeated",
                disambiguated_sentence=_NAPOLEON_SOURCE,
                original_sentence=_NAPOLEON_SOURCE,
            )
        ],
    )

    result = await fidelity_node(state)

    assert [claim.claim_text for claim in result["potential_claims"]] == [_NAPOLEON_SOURCE]
    assert result["potential_claims"][0].fidelity_status == "fallback"


def test_programmatic_fidelity_marks_contextual_additions_borderline() -> None:
    result = assess_claim_fidelity(
        claim_text="Ada Lovelace wrote notes.",
        source_sentence="She wrote notes.",
        context_text="Ada Lovelace wrote notes.",
    )

    assert result.decision == FidelityDecision.BORDERLINE
    assert result.extra_terms == {"ada", "lovelace"}


async def test_fidelity_node_falls_back_to_disambiguated_sentence_when_all_claims_drift(
    monkeypatch,
) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM audit should not run for clear fidelity failures")

    monkeypatch.setattr(fidelity, "_audit_claim_fidelity", fail_if_called)
    state = ExtractorState(
        raw_input="The pyramids were built by aliens.",
        potential_claims=[
            _potential_claim(
                "The pyramids were built by humans.",
                disambiguated_sentence="The pyramids were built by aliens.",
            ),
            _potential_claim(
                "The pyramids were built by ancient Egyptians.",
                disambiguated_sentence="The pyramids were built by aliens.",
            ),
        ],
    )

    result = await fidelity_node(state)

    assert [claim.claim_text for claim in result["potential_claims"]] == [
        "The pyramids were built by aliens."
    ]
    assert result["potential_claims"][0].fidelity_status == "fallback"


async def test_fidelity_node_uses_llm_audit_for_contextual_borderline(monkeypatch) -> None:
    audited_claims: list[str] = []

    async def fake_audit(potential_claim: PotentialClaim) -> bool:
        audited_claims.append(potential_claim.claim_text)
        return True

    monkeypatch.setattr(fidelity, "_audit_claim_fidelity", fake_audit)
    state = ExtractorState(
        raw_input="Ada Lovelace wrote notes.",
        potential_claims=[
            _potential_claim(
                "Ada Lovelace wrote notes.",
                disambiguated_sentence="She wrote notes.",
                original_sentence="Ada Lovelace wrote notes.",
            )
        ],
    )

    result = await fidelity_node(state)

    assert audited_claims == ["Ada Lovelace wrote notes."]
    assert result["potential_claims"][0].fidelity_status == "faithful"


async def test_fidelity_audit_output_normalizes_reasoning_lists_to_strings() -> None:
    output = FidelityAuditOutput(reasoning=["step one", "step two"], faithful=True)

    assert output.reasoning == "step one\nstep two"


async def test_validation_keeps_complete_false_claim_when_llm_rejects_it(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        assert output_class is ValidationOutput
        return ValidationOutput(
            is_complete_declarative=False,
            reasoning="Truth-biased model incorrectly rejects the sentence.",
        )

    monkeypatch.setattr(validation, "call_extractor_structured_output", fake_structured_call)
    monkeypatch.setattr(validation, "get_extractor_llm", lambda **kwargs: object())

    state = ExtractorState(
        raw_input="The pyramids were built by aliens",
        potential_claims=[
            _potential_claim(
                "The pyramids were built by aliens",
                disambiguated_sentence="The pyramids were built by aliens",
            )
        ],
    )

    result = await validation_node(state)

    assert [claim.claim_text for claim in result["validated_claims"]] == [
        "The pyramids were built by aliens"
    ]


def test_verifier_query_prompt_includes_source_assertion() -> None:
    state = VerifierState(
        claim_text="The pyramids were built by aliens.",
        source_sentence="The pyramids were built by aliens.",
    )

    human_prompt = _query_messages(state)[-1][1]

    assert "Source assertion:\nThe pyramids were built by aliens." in human_prompt
    assert "Claim to verify:\nThe pyramids were built by aliens." in human_prompt


def test_verifier_evaluator_prompt_includes_source_assertion() -> None:
    state = VerifierState(
        claim_text="The pyramids were built by aliens.",
        source_sentence="The pyramids were built by aliens.",
        evidence=[
            EvidenceItem(
                url="https://example.com/pyramids",
                title="Pyramids",
                snippet="Reliable sources describe human construction of the pyramids.",
            )
        ],
    )

    human_prompt = _evaluation_messages(state)[-1][1]

    assert "Source assertion:\nThe pyramids were built by aliens." in human_prompt
    assert "Claim to verify:\nThe pyramids were built by aliens." in human_prompt
