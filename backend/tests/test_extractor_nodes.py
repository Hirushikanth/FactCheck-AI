from __future__ import annotations

from factcheck.extractor.nodes import disambiguation, selection, validation
from factcheck.extractor.nodes.decomposition import DecompositionOutput
from factcheck.extractor.nodes.disambiguation import (
    DisambiguationOutput,
    _single_disambiguation_attempt,
    _needs_contextual_disambiguation,
)
from factcheck.extractor.nodes.sentence_splitter import (
    _sentence_splitter_and_context_creator,
    sentence_splitter_node,
)
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


async def test_sentence_splitter_preserves_short_factual_responses() -> None:
    items = await _sentence_splitter_and_context_creator(
        "No. The company denied the claims. Yes. It issued a statement.",
        p_sentences=0,
        f_sentences=0,
    )

    assert [item.original_sentence for item in items] == [
        "No.",
        "The company denied the claims.",
        "Yes.",
        "It issued a statement.",
    ]


async def test_sentence_splitter_merges_non_assertion_fragments() -> None:
    items = await _sentence_splitter_and_context_creator(
        "Based in U.S. The economy grew rapidly.",
        p_sentences=0,
        f_sentences=0,
    )

    assert [item.original_sentence for item in items] == [
        "Based in U.S. The economy grew rapidly.",
    ]


async def test_sentence_splitter_node_builds_per_stage_windows() -> None:
    result = await sentence_splitter_node(
        ExtractorState(
            raw_input=(
                "Ada wrote the first algorithm. "
                "She worked with Charles Babbage. "
                "It was published later."
            )
        )
    )

    selection_items = result["contextual_sentences"]
    preceding_items = result["preceding_context_sentences"]
    middle_selection = selection_items[1]
    middle_preceding = preceding_items[1]

    assert middle_selection.original_sentence == "She worked with Charles Babbage."
    assert middle_preceding.original_sentence == "She worked with Charles Babbage."
    assert "[Following Sentences:]" in middle_selection.context_for_llm
    assert "It was published later." in middle_selection.context_for_llm
    assert "[Following Sentences:]" not in middle_preceding.context_for_llm
    assert "It was published later." not in middle_preceding.context_for_llm


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
                claim_text="ada lovelace wrote notes about the analytical engine.",
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


async def test_validation_node_skips_llm_for_obviously_complete_claim(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("validation LLM should be skipped for obvious declarative claims")

    monkeypatch.setattr(validation, "call_llm_with_structured_output", fail_if_called)
    monkeypatch.setattr(validation, "get_extractor_llm", lambda temperature: object())

    state = ExtractorState(
        raw_input="",
        potential_claims=[
            PotentialClaim(
                claim_text="Ada Lovelace wrote notes about the Analytical Engine.",
                disambiguated_sentence="Ada Lovelace wrote notes about the Analytical Engine.",
                original_sentence="Ada wrote notes about it.",
                original_index=0,
            )
        ],
    )

    result = await validation_node(state)

    assert [claim.claim_text for claim in result["validated_claims"]] == [
        "Ada Lovelace wrote notes about the Analytical Engine."
    ]


async def test_selection_node_uses_batch_voting_and_preserves_context(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("selection_node should not use per-sentence voting")

    async def fake_batch_voting(
        *,
        items,
        batch_processor,
        llm,
        completions,
        min_successes,
        result_factory,
        item_key=None,
    ):
        assert [item.original_index for item in items] == [0, 1]
        assert completions == 3
        assert min_successes == 2
        return [
            result_factory("Ada wrote the first algorithm.", items[0]),
            result_factory("Charles Babbage designed the engine.", items[1]),
        ]

    monkeypatch.setattr(selection, "process_with_voting", fail_if_called)
    monkeypatch.setattr(selection, "process_batch_with_voting", fake_batch_voting, raising=False)
    monkeypatch.setattr(selection, "get_extractor_llm", lambda temperature: object())

    first = ContextualSentence(
        original_sentence="Ada wrote the first algorithm.",
        context_for_llm="[Sentence of Interest for current task:]\nAda wrote the first algorithm.",
        original_index=0,
    )
    second = ContextualSentence(
        original_sentence="He designed the engine.",
        context_for_llm="[Sentence of Interest for current task:]\nHe designed the engine.",
        original_index=1,
    )
    preceding_second = ContextualSentence(
        original_sentence="Charles Babbage designed the engine.",
        context_for_llm="[Preceding Sentences:]\nCharles Babbage designed the engine.",
        original_index=1,
    )

    result = await selection.selection_node(
        ExtractorState(
            raw_input="",
            contextual_sentences=[first, second],
            preceding_context_sentences=[first, preceding_second],
        )
    )

    assert [item.processed_sentence for item in result["selected_contents"]] == [
        "Ada wrote the first algorithm.",
        "Charles Babbage designed the engine.",
    ]
    assert result["selected_contents"][1].preceding_context_item is preceding_second
    assert result["selected_contents"][1].original_context_item is second


async def test_selection_node_reduces_structured_calls_via_batching(monkeypatch) -> None:
    call_prompts: list[str] = []
    responses = [
        selection.BatchSelectionOutput(
            results=[
                selection.BatchSelectionItemOutput(
                    original_index=0,
                    reasoning="first vote",
                    processed_sentence="Ada wrote the first algorithm.",
                    no_verifiable_claims=False,
                    remains_unchanged=True,
                ),
                selection.BatchSelectionItemOutput(
                    original_index=1,
                    reasoning="first vote",
                    processed_sentence="He designed the engine.",
                    no_verifiable_claims=False,
                    remains_unchanged=True,
                ),
            ]
        ),
        selection.BatchSelectionOutput(
            results=[
                selection.BatchSelectionItemOutput(
                    original_index=0,
                    reasoning="second vote",
                    processed_sentence="Ada wrote the first algorithm.",
                    no_verifiable_claims=False,
                    remains_unchanged=True,
                ),
                selection.BatchSelectionItemOutput(
                    original_index=1,
                    reasoning="second vote",
                    processed_sentence="Charles Babbage designed the engine.",
                    no_verifiable_claims=False,
                    remains_unchanged=False,
                ),
            ]
        ),
        selection.BatchSelectionOutput(
            results=[
                selection.BatchSelectionItemOutput(
                    original_index=1,
                    reasoning="third vote",
                    processed_sentence="Charles Babbage designed the engine.",
                    no_verifiable_claims=False,
                    remains_unchanged=False,
                )
            ]
        ),
    ]

    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        call_prompts.append(messages[-1][1])
        return responses[len(call_prompts) - 1]

    monkeypatch.setattr(selection, "call_llm_with_structured_output", fake_structured_call)
    monkeypatch.setattr(selection, "get_extractor_llm", lambda temperature: object())

    first = ContextualSentence(
        original_sentence="Ada wrote the first algorithm.",
        context_for_llm="[Sentence of Interest for current task:]\nAda wrote the first algorithm.",
        original_index=0,
    )
    second = ContextualSentence(
        original_sentence="He designed the engine.",
        context_for_llm="[Sentence of Interest for current task:]\nHe designed the engine.",
        original_index=1,
    )

    result = await selection.selection_node(
        ExtractorState(
            raw_input="",
            contextual_sentences=[first, second],
            preceding_context_sentences=[first, second],
        )
    )

    assert len(call_prompts) == 3
    assert "Sentence #0" in call_prompts[0] and "Sentence #1" in call_prompts[0]
    assert "Sentence #0" in call_prompts[1] and "Sentence #1" in call_prompts[1]
    assert "Sentence #0" not in call_prompts[2] and "Sentence #1" in call_prompts[2]
    assert [item.processed_sentence for item in result["selected_contents"]] == [
        "Ada wrote the first algorithm.",
        "Charles Babbage designed the engine.",
    ]


async def test_selection_node_falls_back_to_legacy_voting_when_batch_fails(monkeypatch) -> None:
    async def fake_batch_voting(*args, **kwargs):
        return []

    async def fake_legacy_voting(
        *,
        items,
        processor,
        llm,
        completions,
        min_successes,
        result_factory,
    ):
        assert [item.original_index for item in items] == [0]
        return [result_factory("Ada wrote the first algorithm.", items[0])]

    monkeypatch.setattr(selection, "process_batch_with_voting", fake_batch_voting)
    monkeypatch.setattr(selection, "process_with_voting", fake_legacy_voting)
    monkeypatch.setattr(selection, "get_extractor_llm", lambda temperature: object())

    item = ContextualSentence(
        original_sentence="Ada wrote the first algorithm.",
        context_for_llm="[Sentence of Interest for current task:]\nAda wrote the first algorithm.",
        original_index=0,
    )

    result = await selection.selection_node(
        ExtractorState(
            raw_input="",
            contextual_sentences=[item],
            preceding_context_sentences=[item],
        )
    )

    assert [selected.processed_sentence for selected in result["selected_contents"]] == [
        "Ada wrote the first algorithm."
    ]


async def test_disambiguation_node_batches_only_contextual_items(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("disambiguation_node should not use per-sentence voting")

    async def fake_batch_voting(
        *,
        items,
        batch_processor,
        llm,
        completions,
        min_successes,
        result_factory,
        item_key=None,
    ):
        assert [item.processed_sentence for item in items] == ["She designed it in 2010."]
        assert completions == 3
        assert min_successes == 2
        return [result_factory("Ada designed the Analytical Engine in 2010.", items[0])]

    monkeypatch.setattr(disambiguation, "process_with_voting", fail_if_called, raising=False)
    monkeypatch.setattr(
        disambiguation, "process_batch_with_voting", fake_batch_voting, raising=False
    )
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    self_contained_context = ContextualSentence(
        original_sentence="Ada Lovelace wrote notes about the Analytical Engine.",
        context_for_llm="[Sentence of Interest for current task:]\nAda Lovelace wrote notes about the Analytical Engine.",
        original_index=0,
    )
    ambiguous_context = ContextualSentence(
        original_sentence="Ada designed the Analytical Engine in 2010.",
        context_for_llm="[Preceding Sentences:]\nAda designed the Analytical Engine in 2010.",
        original_index=1,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="Ada Lovelace wrote notes about the Analytical Engine.",
                    original_context_item=self_contained_context,
                    preceding_context_item=self_contained_context,
                ),
                SelectedContent(
                    processed_sentence="She designed it in 2010.",
                    original_context_item=ambiguous_context,
                    preceding_context_item=ambiguous_context,
                ),
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "Ada Lovelace wrote notes about the Analytical Engine.",
        "Ada designed the Analytical Engine in 2010.",
    ]
    assert result["disambiguated_contents"][0].original_selected_item.processed_sentence == (
        "Ada Lovelace wrote notes about the Analytical Engine."
    )


async def test_disambiguation_node_falls_back_to_legacy_voting_when_batch_fails(
    monkeypatch,
) -> None:
    async def fake_batch_voting(*args, **kwargs):
        return []

    async def fake_legacy_voting(
        *,
        items,
        processor,
        llm,
        completions,
        min_successes,
        result_factory,
    ):
        assert [item.processed_sentence for item in items] == ["Smith led the operations team."]
        return [result_factory("John Smith led the operations team.", items[0])]

    monkeypatch.setattr(disambiguation, "process_batch_with_voting", fake_batch_voting)
    monkeypatch.setattr(disambiguation, "process_with_voting", fake_legacy_voting, raising=False)
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    context_item = ContextualSentence(
        original_sentence="John Smith transitioned to management in 2010.",
        context_for_llm="[Preceding Sentences:]\nJohn Smith transitioned to management in 2010.",
        original_index=0,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="Smith led the operations team.",
                    original_context_item=context_item,
                    preceding_context_item=context_item,
                )
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "John Smith led the operations team."
    ]


async def test_disambiguation_node_batches_partial_name_cases(monkeypatch) -> None:
    async def fake_batch_voting(
        *,
        items,
        batch_processor,
        llm,
        completions,
        min_successes,
        result_factory,
        item_key=None,
    ):
        assert [item.processed_sentence for item in items] == ["Smith led the operations team."]
        return [result_factory("John Smith led the operations team.", items[0])]

    monkeypatch.setattr(
        disambiguation, "process_batch_with_voting", fake_batch_voting, raising=False
    )
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    context_item = ContextualSentence(
        original_sentence="John Smith transitioned to management in 2010.",
        context_for_llm="[Preceding Sentences:]\nJohn Smith transitioned to management in 2010.",
        original_index=0,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="Smith led the operations team.",
                    original_context_item=context_item,
                    preceding_context_item=context_item,
                )
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "John Smith led the operations team."
    ]


async def test_disambiguation_node_batches_first_name_only_cases(monkeypatch) -> None:
    async def fake_batch_voting(
        *,
        items,
        batch_processor,
        llm,
        completions,
        min_successes,
        result_factory,
        item_key=None,
    ):
        assert [item.processed_sentence for item in items] == ["John led the operations team."]
        return [result_factory("John Smith led the operations team.", items[0])]

    monkeypatch.setattr(
        disambiguation, "process_batch_with_voting", fake_batch_voting, raising=False
    )
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    context_item = ContextualSentence(
        original_sentence="John Smith transitioned to management in 2010.",
        context_for_llm="[Preceding Sentences:]\nJohn Smith transitioned to management in 2010.",
        original_index=0,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="John led the operations team.",
                    original_context_item=context_item,
                    preceding_context_item=context_item,
                )
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "John Smith led the operations team."
    ]


async def test_disambiguation_node_skips_self_contained_acronym_without_expansion(
    monkeypatch,
) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("self-contained acronym sentence should not be batched")

    monkeypatch.setattr(
        disambiguation, "process_batch_with_voting", fail_if_called, raising=False
    )
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    context_item = ContextualSentence(
        original_sentence="WHO released a report.",
        context_for_llm="[Sentence of Interest for current task:]\nWHO released a report.",
        original_index=0,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="WHO released a report.",
                    original_context_item=context_item,
                    preceding_context_item=context_item,
                )
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "WHO released a report."
    ]


async def test_disambiguation_node_batches_non_parenthetical_acronym_expansion(
    monkeypatch,
) -> None:
    async def fake_batch_voting(
        *,
        items,
        batch_processor,
        llm,
        completions,
        min_successes,
        result_factory,
        item_key=None,
    ):
        assert [item.processed_sentence for item in items] == ["WHO released a report."]
        return [result_factory("World Health Organization released a report.", items[0])]

    monkeypatch.setattr(
        disambiguation, "process_batch_with_voting", fake_batch_voting, raising=False
    )
    monkeypatch.setattr(disambiguation, "get_extractor_llm", lambda temperature: object())

    context_item = ContextualSentence(
        original_sentence="World Health Organization, WHO, released a report.",
        context_for_llm=(
            "[Preceding Sentences:]\nWorld Health Organization, WHO, released a report."
        ),
        original_index=0,
    )

    result = await disambiguation.disambiguation_node(
        ExtractorState(
            raw_input="",
            selected_contents=[
                SelectedContent(
                    processed_sentence="WHO released a report.",
                    original_context_item=context_item,
                    preceding_context_item=context_item,
                )
            ],
        )
    )

    assert [item.disambiguated_sentence for item in result["disambiguated_contents"]] == [
        "World Health Organization released a report."
    ]


def test_disambiguation_reference_heuristic_detects_contextual_references() -> None:
    assert not _needs_contextual_disambiguation(
        "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
    )
    assert not _needs_contextual_disambiguation(
        "There is strong evidence of vaccine effectiveness."
    )
    assert not _needs_contextual_disambiguation(
        "There are multiple published studies in the literature."
    )
    assert not _needs_contextual_disambiguation(
        "Here, the researchers found significant results."
    )
    assert _needs_contextual_disambiguation("She wrote notes about it.")
    assert _needs_contextual_disambiguation("The experiments were conducted there.")


def test_validation_heuristic_rejects_plural_noun_fragments() -> None:
    assert not validation._looks_like_complete_declarative("Company earnings results")


def test_validation_heuristic_rejects_ed_noun_phrase_fragments() -> None:
    assert not validation._looks_like_complete_declarative("Company-funded vaccine study")
    assert not validation._looks_like_complete_declarative("A government-backed clean energy plan")


async def test_disambiguation_skips_existential_there(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for existential there")

    monkeypatch.setattr(disambiguation, "call_llm_with_structured_output", fail_if_called)
    preceding_context_item = ContextualSentence(
        original_sentence="There is strong evidence of vaccine effectiveness.",
        context_for_llm=(
            "[Sentence of Interest for current task:]\n"
            "There is strong evidence of vaccine effectiveness."
        ),
        original_index=0,
    )
    selected = SelectedContent(
        processed_sentence="There is strong evidence of vaccine effectiveness.",
        original_context_item=preceding_context_item,
        preceding_context_item=preceding_context_item,
    )

    success, sentence = await _single_disambiguation_attempt(selected, object())

    assert success is True
    assert sentence == "There is strong evidence of vaccine effectiveness."


async def test_disambiguation_passes_through_self_contained_sentence(monkeypatch) -> None:
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for self-contained sentence")

    monkeypatch.setattr(disambiguation, "call_llm_with_structured_output", fail_if_called)
    preceding_context_item = ContextualSentence(
        original_sentence="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
        context_for_llm=(
            "[Sentence of Interest for current task:]\n"
            "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
        ),
        original_index=0,
    )
    selected = SelectedContent(
        processed_sentence="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
        original_context_item=preceding_context_item,
        preceding_context_item=preceding_context_item,
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
