from __future__ import annotations

from factcheck.extractor.nodes import disambiguation, validation
from factcheck.extractor.nodes.decomposition import DecompositionOutput
from factcheck.extractor.nodes.disambiguation import (
    DisambiguationOutput,
    _single_disambiguation_attempt,
    _needs_contextual_disambiguation,
)
from factcheck.extractor.nodes.sentence_splitter import _sentence_splitter_and_context_creator
from factcheck.extractor.nodes.selection import SelectionOutput
from factcheck.extractor.nodes.validation import ValidationOutput, validation_node
from factcheck.extractor.schemas import ContextualSentence, ExtractorState, PotentialClaim, SelectedContent


def test_extractor_outputs_start_with_reasoning_field() -> None:
    assert list(ValidationOutput.model_fields) == ["reasoning", "is_complete_declarative"]
    assert list(SelectionOutput.model_fields) == [
        "reasoning",
        "processed_sentence",
        "no_verifiable_claims",
        "remains_unchanged",
    ]
    assert list(DisambiguationOutput.model_fields) == [
        "reasoning",
        "disambiguated_sentence",
        "cannot_be_disambiguated",
    ]
    assert list(DecompositionOutput.model_fields) == ["reasoning", "claims", "no_claims"]


def test_extractor_outputs_normalize_reasoning_lists_to_strings() -> None:
    expected_reasoning = "step one\nstep two"

    outputs = [
        ValidationOutput(
            reasoning=["step one", "step two"],
            is_complete_declarative=True,
        ),
        SelectionOutput(
            reasoning=["step one", "step two"],
            processed_sentence="Ada Lovelace wrote notes.",
            no_verifiable_claims=False,
            remains_unchanged=True,
        ),
        DisambiguationOutput(
            reasoning=["step one", "step two"],
            disambiguated_sentence="Ada Lovelace wrote notes.",
            cannot_be_disambiguated=False,
        ),
        DecompositionOutput(
            reasoning=["step one", "step two"],
            claims=["Ada Lovelace wrote notes."],
            no_claims=False,
        ),
    ]

    assert [output.reasoning for output in outputs] == [expected_reasoning] * len(outputs)


async def test_sentence_splitter_builds_context_windows() -> None:
    items = await _sentence_splitter_and_context_creator(
        "Ada wrote the first algorithm. She worked with Charles Babbage. It was published later.",
        p_sentences=1,
        f_sentences=1,
    )

    assert [item.original_sentence for item in items] == [
        "Ada wrote the first algorithm.",
        "She worked with Charles Babbage.",
        "It was published later.",
    ]
    assert "[Preceding Sentences:]" in items[1].context_for_llm
    assert "Ada wrote the first algorithm." in items[1].context_for_llm
    assert "[Following Sentences:]" in items[1].context_for_llm
    assert "It was published later." in items[1].context_for_llm


async def test_validation_node_filters_invalid_and_duplicate_claims(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        claim_text = messages[-1][1]
        return ValidationOutput(
            reasoning="Test stub reasoning.",
            is_complete_declarative="Fragment" not in claim_text,
        )

    monkeypatch.setattr(validation, "call_llm_with_structured_output", fake_structured_call)
    monkeypatch.setattr(validation, "get_extractor_llm", lambda temperature: object())

    state = ExtractorState(
        raw_input="",
        potential_claims=[
            PotentialClaim(
                claim_text="Ada Lovelace wrote notes about the Analytical Engine.",
                disambiguated_sentence="Ada Lovelace wrote notes about the Analytical Engine.",
                original_sentence="Ada wrote notes about it.",
                original_index=0,
            ),
            PotentialClaim(
                claim_text="Ada Lovelace wrote notes about the Analytical Engine.",
                disambiguated_sentence="Ada Lovelace wrote notes about the Analytical Engine.",
                original_sentence="Ada wrote notes about it.",
                original_index=0,
            ),
            PotentialClaim(
                claim_text="Fragment about an engine",
                disambiguated_sentence="Fragment about an engine",
                original_sentence="Fragment about it.",
                original_index=1,
            ),
        ],
    )

    result = await validation_node(state)

    assert [claim.claim_text for claim in result["validated_claims"]] == [
        "Ada Lovelace wrote notes about the Analytical Engine."
    ]


def test_disambiguation_reference_heuristic_detects_contextual_references() -> None:
    assert not _needs_contextual_disambiguation(
        "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
    )
    assert _needs_contextual_disambiguation("She wrote notes about it.")


async def test_disambiguation_passes_through_self_contained_sentence(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for self-contained sentence")

    monkeypatch.setattr(disambiguation, "call_llm_with_structured_output", fail_if_called)
    selected = SelectedContent(
        processed_sentence="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
        original_context_item=ContextualSentence(
            original_sentence="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
            context_for_llm=(
                "[Sentence of Interest for current task:]\n"
                "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
            ),
            original_index=0,
        ),
    )

    success, sentence = await _single_disambiguation_attempt(selected, object())

    assert success is True
    assert sentence == "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."


async def test_pysbd_sentence_splitting_handles_complex_abbreviations() -> None:
    text = "Dr. Smith works at OpenAI Inc. in London. He is a senior scientist."
    items = await _sentence_splitter_and_context_creator(text, p_sentences=0, f_sentences=0)

    assert [item.original_sentence for item in items] == [
        "Dr. Smith works at OpenAI Inc. in London.",
        "He is a senior scientist.",
    ]


async def test_pysbd_sentence_splitting_handles_washington_dc() -> None:
    text = "Dr. Smith works at OpenAI Inc. in Washington D.C. He joined in 2024."
    items = await _sentence_splitter_and_context_creator(text, p_sentences=0, f_sentences=0)

    assert [item.original_sentence for item in items] == [
        "Dr. Smith works at OpenAI Inc. in Washington D.C.",
        "He joined in 2024.",
    ]


async def test_pysbd_sentence_splitting_handles_washington_dc_before_acronym() -> None:
    text = "Dr. Smith works at OpenAI Inc. in Washington D.C. USA citizens voted in November."
    items = await _sentence_splitter_and_context_creator(text, p_sentences=0, f_sentences=0)

    assert [item.original_sentence for item in items] == [
        "Dr. Smith works at OpenAI Inc. in Washington D.C.",
        "USA citizens voted in November.",
    ]


async def test_pysbd_sentence_splitting_handles_washington_dc_before_year() -> None:
    text = "Dr. Smith works at OpenAI Inc. in Washington D.C. 2026 was a turning point for policy."
    items = await _sentence_splitter_and_context_creator(text, p_sentences=0, f_sentences=0)

    assert [item.original_sentence for item in items] == [
        "Dr. Smith works at OpenAI Inc. in Washington D.C.",
        "2026 was a turning point for policy.",
    ]
