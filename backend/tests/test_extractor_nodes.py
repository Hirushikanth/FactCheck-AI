from __future__ import annotations

from factcheck.extractor.nodes import disambiguation, sentence_splitter, validation
from factcheck.extractor.nodes.disambiguation import (
    _single_disambiguation_attempt,
    _needs_contextual_disambiguation,
)
from factcheck.extractor.nodes.sentence_splitter import _sentence_splitter_and_context_creator
from factcheck.extractor.nodes.validation import ValidationOutput, validation_node
from factcheck.extractor.schemas import ContextualSentence, ExtractorState, PotentialClaim, SelectedContent


async def test_sentence_splitter_builds_context_windows(monkeypatch) -> None:
    monkeypatch.setattr(sentence_splitter, "ensure_nltk_resources", lambda: None)
    monkeypatch.setattr(
        sentence_splitter.nltk,
        "sent_tokenize",
        lambda paragraph: [part.strip() + "." for part in paragraph.split(".") if part.strip()],
    )

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


def test_ensure_nltk_resources_downloads_each_missing_resource(monkeypatch) -> None:
    downloads: list[str] = []

    def fake_find(resource: str):
        if resource == "tokenizers/punkt":
            return object()
        raise LookupError(resource)

    monkeypatch.setattr(sentence_splitter.nltk.data, "find", fake_find)
    monkeypatch.setattr(
        sentence_splitter.nltk,
        "download",
        lambda resource, quiet: downloads.append(resource),
    )

    sentence_splitter.ensure_nltk_resources()

    assert downloads == ["punkt_tab"]


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
