from __future__ import annotations

import pytest

from factcheck.extractor.nodes import selection
from factcheck.extractor.nodes.sentence_splitter import sentence_splitter_node
from factcheck.extractor.schemas import ExtractorState


@pytest.mark.parametrize(
    "raw_input",
    [
        "bats are blind",
        "The Great Wall of China is visible from space with the naked eye.",
        "Lightning never strikes the same place twice.",
    ],
)
async def test_selection_skips_llm_for_direct_claim_inputs(
    raw_input: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("selection LLM should not be called for direct_claim input")

    monkeypatch.setattr(selection, "call_extractor_structured_output", fail_if_called)
    monkeypatch.setattr(selection, "get_extractor_llm", lambda **kwargs: object())

    split = await sentence_splitter_node(ExtractorState(raw_input=raw_input))
    state = ExtractorState(
        raw_input=raw_input,
        contextual_sentences=split["contextual_sentences"],
        preceding_context_sentences=split["preceding_context_sentences"],
    )

    result = await selection.selection_node(state)

    assert result["resolved_extraction_mode"] == "direct_claim"
    assert len(result["selected_contents"]) == 1
    assert (
        result["selected_contents"][0].processed_sentence
        == split["contextual_sentences"][0].original_sentence
    )


async def test_selection_uses_llm_for_multi_sentence_document(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        calls.append(messages[-1][1])
        sentence = messages[-1][1].split("Sentence:\n")[-1].strip()
        return selection.SelectionOutput(
            no_verifiable_claims=False,
            remains_unchanged=True,
            processed_sentence=sentence,
            reasoning="Test stub.",
        )

    monkeypatch.setattr(selection, "call_extractor_structured_output", fake_structured_call)
    monkeypatch.setattr(selection, "get_extractor_llm", lambda **kwargs: object())

    raw_input = (
        "Many myths persist today. "
        "The Great Wall of China is visible from space with the naked eye."
    )
    split = await sentence_splitter_node(ExtractorState(raw_input=raw_input))
    state = ExtractorState(
        raw_input=raw_input,
        contextual_sentences=split["contextual_sentences"],
        preceding_context_sentences=split["preceding_context_sentences"],
    )

    result = await selection.selection_node(state)

    assert result["resolved_extraction_mode"] == "document"
    assert calls
