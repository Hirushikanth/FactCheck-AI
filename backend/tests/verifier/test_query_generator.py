from __future__ import annotations

from factcheck.verifier.nodes import query_generator
from factcheck.verifier.nodes.query_generator import (
    QueryGeneratorOutput,
    _clean_queries,
    _clean_query,
    query_generator_node,
)
from factcheck.verifier.schemas import IntermediateAssessment, VerifierState


async def test_query_generator_sets_current_query_and_appends_history(monkeypatch) -> None:
    llm_calls: list[dict[str, object]] = []

    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        assert output_class is QueryGeneratorOutput
        return QueryGeneratorOutput(queries=["The Earth is an oblate spheroid."])

    def fake_get_verifier_llm(**kwargs):
        llm_calls.append(kwargs)
        return object()

    monkeypatch.setattr(query_generator, "get_verifier_llm", fake_get_verifier_llm)
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(VerifierState(claim_text="The Earth is an oblate spheroid."))

    assert result["current_query"] == "The Earth is an oblate spheroid"
    assert result["current_queries"] == [
        "The Earth is an oblate spheroid",
        "Earth oblate spheroid",
    ]
    assert result["all_queries"] == [
        "The Earth is an oblate spheroid",
        "Earth oblate spheroid",
    ]
    assert llm_calls == [{"temperature": 0.0, "num_ctx": 2048}]


async def test_query_generator_returns_multiple_initial_queries(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return QueryGeneratorOutput(
            queries=[
                "NASA Earth oblate spheroid official",
                "Earth shape fact check scientific source",
            ]
        )

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(VerifierState(claim_text="The Earth is an oblate spheroid."))

    assert result["current_queries"] == [
        "NASA Earth oblate spheroid official",
        "Earth shape fact check scientific source",
    ]
    assert result["current_query"] == "NASA Earth oblate spheroid official"
    assert result["all_queries"] == result["current_queries"]


async def test_query_generator_uses_missing_aspects_for_iterative_queries(monkeypatch) -> None:
    captured_messages = []

    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        captured_messages.extend(messages)
        return QueryGeneratorOutput(queries=["NASA Earth shape independent source"])

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            all_queries=["Earth oblate spheroid fact check"],
            iteration_count=1,
            intermediate_assessment=IntermediateAssessment(
                needs_more_evidence=True,
                missing_aspects=["independent scientific source"],
            ),
        )
    )

    human_prompt = captured_messages[-1][1]
    assert "Earth oblate spheroid fact check" in human_prompt
    assert "independent scientific source" in human_prompt
    assert result["current_query"] == "NASA Earth shape independent source"
    assert result["current_queries"] == ["NASA Earth shape independent source"]
    assert result["all_queries"] == [
        "Earth oblate spheroid fact check",
        "NASA Earth shape independent source",
    ]


async def test_query_generator_keyword_fallback_when_llm_returns_literal_claim(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return QueryGeneratorOutput(queries=["The Earth is an oblate spheroid."])

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            all_queries=["The Earth is an oblate spheroid"],
        )
    )

    assert result["current_query"] == "Earth oblate spheroid"
    assert result["current_queries"] == [
        "Earth oblate spheroid",
        "The Earth is an oblate spheroid fact check",
    ]
    assert result["all_queries"] == [
        "The Earth is an oblate spheroid",
        "Earth oblate spheroid",
        "The Earth is an oblate spheroid fact check",
    ]


def test_clean_query_returns_none_when_all_fallbacks_exhausted() -> None:
    claim = "The Earth is an oblate spheroid."
    previous_queries = [
        "Earth oblate spheroid",
        "The Earth is an oblate spheroid fact check",
        "The Earth is an oblate spheroid",
    ]

    assert _clean_query([], claim, previous_queries) is None
    assert _clean_queries([], claim, previous_queries, max_queries=2) == []


def test_clean_query_uses_missing_aspect_fallback() -> None:
    claim = "The Earth is an oblate spheroid."
    previous_queries = [
        "Earth oblate spheroid",
        "The Earth is an oblate spheroid fact check",
        "The Earth is an oblate spheroid",
    ]

    query = _clean_query(
        [],
        claim,
        previous_queries,
        missing_aspects=["independent scientific source"],
    )

    assert query == "independent scientific source The Earth is an oblate spheroid"


async def test_query_generator_sets_search_exhausted_when_query_budget_exhausted(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return QueryGeneratorOutput(queries=["new query attempt"])

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            all_queries=[f"query {index}" for index in range(10)],
            iteration_count=1,
        )
    )

    assert result["current_query"] is None
    assert result["current_queries"] == []
    assert result["search_exhausted"] is True


async def test_query_generator_sets_search_exhausted_on_duplicate_query(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return QueryGeneratorOutput(queries=["The Earth is an oblate spheroid fact check"])

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda **kwargs: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(
        VerifierState(
            claim_text="The Earth is an oblate spheroid.",
            all_queries=[
                "Earth oblate spheroid",
                "The Earth is an oblate spheroid fact check",
                "The Earth is an oblate spheroid",
                "independent scientific source The Earth is an oblate spheroid",
                "independent scientific source",
            ],
            iteration_count=1,
            intermediate_assessment=IntermediateAssessment(
                needs_more_evidence=True,
                missing_aspects=["independent scientific source"],
            ),
        )
    )

    assert result["current_query"] is None
    assert result["current_queries"] == []
    assert result["search_exhausted"] is True
