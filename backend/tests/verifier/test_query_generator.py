from __future__ import annotations

from factcheck.verifier.nodes import query_generator
from factcheck.verifier.nodes.query_generator import QueryGeneratorOutput, query_generator_node
from factcheck.verifier.schemas import VerifierState


async def test_query_generator_expands_literal_claim_query(monkeypatch) -> None:
    async def fake_structured_call(*, llm, output_class, messages, context_desc=""):
        return QueryGeneratorOutput(queries=["The Earth is an oblate spheroid."])

    monkeypatch.setattr(query_generator, "get_verifier_llm", lambda temperature: object())
    monkeypatch.setattr(query_generator, "call_llm_with_structured_output", fake_structured_call)

    result = await query_generator_node(VerifierState(claim="The Earth is an oblate spheroid."))

    assert result["search_queries"] == [
        "The Earth is an oblate spheroid",
        "Earth oblate spheroid",
        "The Earth is an oblate spheroid fact check",
    ]
