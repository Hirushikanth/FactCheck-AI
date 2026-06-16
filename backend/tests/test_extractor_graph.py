from __future__ import annotations

from factcheck.extractor import graph as extractor_graph
from factcheck.extractor import ExtractorRunResult, run_extractor
from factcheck.extractor.graph import build_extractor_graph
from factcheck.extractor.schemas import (
    ContextualSentence,
    DisambiguatedContent,
    ExtractorState,
    PotentialClaim,
    SelectedContent,
    ValidatedClaim,
)


async def test_extractor_graph_runs_claim_extraction_stages_in_order(monkeypatch) -> None:
    calls: list[str] = []

    async def sentence_splitter_node(state):
        calls.append("sentence_splitter")
        contextual = ContextualSentence(
            original_sentence="Ada wrote the first algorithm.",
            context_for_llm="[Sentence of Interest for current task:]\nAda wrote the first algorithm.",
            original_index=0,
        )
        return {
            "contextual_sentences": [contextual],
            "preceding_context_sentences": [contextual],
        }

    async def selection_node(state):
        calls.append("selection")
        return {
            "selected_contents": [
                SelectedContent(
                    processed_sentence=state.contextual_sentences[0].original_sentence,
                    original_context_item=state.contextual_sentences[0],
                    preceding_context_item=state.preceding_context_sentences[0],
                )
            ]
        }

    async def disambiguation_node(state):
        calls.append("disambiguation")
        return {
            "disambiguated_contents": [
                DisambiguatedContent(
                    disambiguated_sentence="Ada Lovelace wrote the first algorithm.",
                    original_selected_item=state.selected_contents[0],
                )
            ]
        }

    async def decomposition_node(state):
        calls.append("decomposition")
        return {
            "potential_claims": [
                PotentialClaim(
                    claim_text="Ada Lovelace wrote the first algorithm.",
                    disambiguated_sentence=state.disambiguated_contents[0].disambiguated_sentence,
                    original_sentence="Ada wrote the first algorithm.",
                    original_index=0,
                )
            ]
        }

    async def validation_node(state):
        calls.append("validation")
        return {
            "validated_claims": [
                ValidatedClaim(
                    claim_text=state.potential_claims[0].claim_text,
                    is_complete_declarative=True,
                    disambiguated_sentence=state.potential_claims[0].disambiguated_sentence,
                    original_sentence=state.potential_claims[0].original_sentence,
                    original_index=0,
                )
            ]
        }

    monkeypatch.setattr(extractor_graph, "sentence_splitter_node", sentence_splitter_node)
    monkeypatch.setattr(extractor_graph, "selection_node", selection_node)
    monkeypatch.setattr(extractor_graph, "disambiguation_node", disambiguation_node)
    monkeypatch.setattr(extractor_graph, "decomposition_node", decomposition_node)
    monkeypatch.setattr(extractor_graph, "validation_node", validation_node)

    graph = build_extractor_graph()
    result = await graph.ainvoke(ExtractorState(raw_input="Ada wrote the first algorithm."))

    assert calls == [
        "sentence_splitter",
        "selection",
        "disambiguation",
        "decomposition",
        "validation",
    ]
    assert result["validated_claims"][0].claim_text == "Ada Lovelace wrote the first algorithm."


async def test_run_extractor_returns_validated_claims(monkeypatch) -> None:
    validated_claim = ValidatedClaim(
        claim_text="Ada Lovelace wrote the first algorithm.",
        is_complete_declarative=True,
        disambiguated_sentence="Ada Lovelace wrote the first algorithm.",
        original_sentence="Ada wrote the first algorithm.",
        original_index=0,
    )

    class FakeExtractorGraph:
        async def ainvoke(self, state):
            assert state.raw_input == "Ada wrote the first algorithm."
            return {"validated_claims": [validated_claim]}

    monkeypatch.setattr(extractor_graph, "build_extractor_graph", lambda: FakeExtractorGraph())

    result = await run_extractor("Ada wrote the first algorithm.")
    assert result.claims == [validated_claim]
    assert result.stage_failures == []


def test_extractor_run_result_is_iterable_over_claims() -> None:
    claim = ValidatedClaim(
        claim_text="Ada Lovelace wrote the first algorithm.",
        is_complete_declarative=True,
        disambiguated_sentence="Ada Lovelace wrote the first algorithm.",
        original_sentence="Ada wrote the first algorithm.",
        original_index=0,
    )
    result = ExtractorRunResult(claims=[claim], stage_failures=[])

    assert list(result) == [claim]
    assert len(result) == 1
