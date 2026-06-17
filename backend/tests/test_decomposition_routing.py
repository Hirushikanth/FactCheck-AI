from __future__ import annotations

import pytest

from factcheck.extractor.nodes import decomposition
from factcheck.extractor.schemas import (
    ContextualSentence,
    DisambiguatedContent,
    ExtractorState,
    SelectedContent,
)


def _disambiguated_item(sentence: str) -> DisambiguatedContent:
    contextual = ContextualSentence(
        original_sentence=sentence,
        context_for_llm=f"[Sentence of Interest for current task:]\n{sentence}",
        original_index=0,
    )
    selected = SelectedContent(
        processed_sentence=sentence,
        original_context_item=contextual,
        preceding_context_item=contextual,
    )
    return DisambiguatedContent(
        disambiguated_sentence=sentence,
        original_selected_item=selected,
    )


async def test_decomposition_skips_llm_for_simple_direct_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("decomposition LLM should not be called for simple direct claim")

    monkeypatch.setattr(decomposition, "call_extractor_structured_output", fail_if_called)
    monkeypatch.setattr(decomposition, "get_extractor_llm", lambda **kwargs: object())

    sentence = "Lightning never strikes the same place twice."
    state = ExtractorState(
        raw_input=sentence,
        resolved_extraction_mode="direct_claim",
        disambiguated_contents=[_disambiguated_item(sentence)],
    )

    result = await decomposition.decomposition_node(state)

    assert len(result["potential_claims"]) == 1
    assert result["potential_claims"][0].claim_text == sentence


async def test_decomposition_calls_llm_for_compound_direct_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        calls.append(context_desc)
        return decomposition.DecompositionOutput(
            no_claims=False,
            claims=[
                "Bananas are berries.",
                "Strawberries are not berries.",
            ],
            reasoning="Test stub.",
        )

    monkeypatch.setattr(decomposition, "call_extractor_structured_output", fake_structured_call)
    monkeypatch.setattr(decomposition, "get_extractor_llm", lambda **kwargs: object())

    sentence = "Bananas are berries, but strawberries are not."
    state = ExtractorState(
        raw_input=sentence,
        resolved_extraction_mode="direct_claim",
        disambiguated_contents=[_disambiguated_item(sentence)],
    )

    result = await decomposition.decomposition_node(state)

    assert calls
    assert len(result["potential_claims"]) == 2
