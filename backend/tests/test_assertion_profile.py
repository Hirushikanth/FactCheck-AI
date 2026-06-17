from __future__ import annotations

import pytest

from factcheck.extractor.utils.assertion_profile import (
    looks_like_complete_declarative,
    profile_sentence,
    resolve_extraction_mode,
    sentence_has_compound_structure,
)


@pytest.mark.parametrize(
    "sentence",
    [
        "bats are blind",
        "The Great Wall of China is visible from space with the naked eye.",
        "Lightning never strikes the same place twice.",
        "Vaccines cause autism.",
        "The Earth is flat.",
        "Drinking bleach cures COVID-19.",
    ],
)
def test_myth_sentences_profile_as_checkable_fact(sentence: str) -> None:
    profile = profile_sentence(sentence)
    assert profile.kind == "checkable_fact"
    assert looks_like_complete_declarative(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Technological progress should be inclusive",
        "The government is terrible",
        "Our product is the best solution for teams",
    ],
)
def test_opinion_sentences_are_not_checkable_fact(sentence: str) -> None:
    profile = profile_sentence(sentence)
    assert profile.kind == "opinion"


def test_anaphoric_sentence_profiles_as_anaphoric() -> None:
    profile = profile_sentence("She wrote notes about it.")
    assert profile.kind == "anaphoric"


def test_hedge_without_anchor_profiles_as_hedge() -> None:
    profile = profile_sentence("AI could lead to advancements")
    assert profile.kind == "hedge"


def test_resolve_auto_direct_claim_for_single_myth() -> None:
    sentence = "Lightning never strikes the same place twice."
    assert resolve_extraction_mode([sentence], forced="auto") == "direct_claim"


def test_resolve_auto_document_for_multiple_sentences() -> None:
    sentences = [
        "Many myths persist today.",
        "The Great Wall of China is visible from space with the naked eye.",
    ]
    assert resolve_extraction_mode(sentences, forced="auto") == "document"


def test_resolve_auto_document_for_single_opinion() -> None:
    assert resolve_extraction_mode(["The government is terrible"], forced="auto") == "document"


def test_resolve_forced_claim_mode() -> None:
    assert (
        resolve_extraction_mode(["The government is terrible"], forced="claim")
        == "direct_claim"
    )


def test_resolve_forced_document_mode() -> None:
    assert (
        resolve_extraction_mode(["Lightning never strikes the same place twice."], forced="document")
        == "document"
    )


def test_compound_structure_detected_for_contrastive_sentence() -> None:
    sentence = "Bananas are berries, but strawberries are not."
    assert sentence_has_compound_structure(sentence)


def test_compound_structure_not_detected_for_simple_myth() -> None:
    sentence = "The Great Wall of China is visible from space with the naked eye."
    assert not sentence_has_compound_structure(sentence)
