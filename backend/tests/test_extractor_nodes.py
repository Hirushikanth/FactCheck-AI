from __future__ import annotations

from factcheck.extractor.nodes import disambiguation, sentence_splitter, validation
from factcheck.extractor.nodes.disambiguation import (
    _single_disambiguation_attempt,
    _needs_contextual_disambiguation,
)
from factcheck.extractor.nodes.sentence_splitter import _sentence_splitter_and_context_creator
from factcheck.extractor.nodes.validation import ValidationOutput, validation_node
from factcheck.extractor.prompts import VALIDATION_SYSTEM_PROMPT
from factcheck.extractor.schemas import ContextualSentence, ExtractorState, PotentialClaim, SelectedContent


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


def test_validation_prompt_matches_structured_output_contract() -> None:
    assert "structured fields" in VALIDATION_SYSTEM_PROMPT
    assert "is_complete_declarative" in VALIDATION_SYSTEM_PROMPT
    assert 'Print "C =' not in VALIDATION_SYSTEM_PROMPT


async def test_validation_node_filters_invalid_and_duplicate_claims(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        claim_text = messages[-1][1]
        return ValidationOutput(is_complete_declarative="Fragment" not in claim_text)

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
