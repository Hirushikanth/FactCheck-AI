from factcheck.graph.pipeline import build_graph
from factcheck.extractor.schemas import ValidatedClaim


def _validated_claim() -> ValidatedClaim:
    return ValidatedClaim(
        claim_text="The Earth is round.",
        is_complete_declarative=True,
        disambiguated_sentence="The Earth is round.",
        original_sentence="The Earth is round.",
        original_index=0,
    )


async def test_pipeline_runs_extractor_before_verifier(monkeypatch) -> None:
    claim = _validated_claim()

    async def extractor_stub(state):
        return {"current_agent": "extractor", "extracted_claims": [claim]}

    async def verifier_stub(verifier_claim: ValidatedClaim):
        assert verifier_claim == claim
        return {
            "claim": verifier_claim.claim_text,
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "evidence": ["Earth is round."],
            "sources": ["https://example.com/earth"],
            "reasoning": "The evidence supports the claim.",
            "search_queries": ["earth round"],
        }

    import factcheck.graph.pipeline as pipeline
    import factcheck.agents.verifier as verifier

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)
    monkeypatch.setattr(verifier, "run_verifier", verifier_stub)

    graph = build_graph()
    result = await graph.ainvoke(
        {
            "raw_input": "The Earth is round.",
            "extracted_claims": [],
            "claim_results": [],
            "final_report": None,
            "messages": [],
            "current_agent": "",
            "session_id": "test-session",
            "error": None,
            "status": "idle",
        }
    )

    assert result["current_agent"] == "reporter"
    assert result["status"] == "done"
    assert result["final_report"] == "Phase 1 pipeline scaffold completed."
    assert result["extracted_claims"] == [claim]
    assert len(result["claim_results"]) == 1
    assert result["claim_results"][0]["verdict"] == "SUPPORTED"


async def test_pipeline_preserves_verifier_error_status(monkeypatch) -> None:
    claim = _validated_claim()

    async def extractor_stub(state):
        return {"current_agent": "extractor", "extracted_claims": [claim]}

    async def verifier_stub(verifier_claim: ValidatedClaim):
        assert verifier_claim == claim
        raise RuntimeError("boom")

    import factcheck.graph.pipeline as pipeline
    import factcheck.agents.verifier as verifier

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)
    monkeypatch.setattr(verifier, "run_verifier", verifier_stub)

    graph = build_graph()
    result = await graph.ainvoke(
        {
            "raw_input": "The Earth is round.",
            "extracted_claims": [],
            "claim_results": [],
            "final_report": None,
            "messages": [],
            "current_agent": "",
            "session_id": "test-session",
            "error": None,
            "status": "idle",
        }
    )

    assert result["current_agent"] == "verifier"
    assert result["status"] == "error"
    assert result["error"] == "boom"
    assert result["final_report"] is None
    assert result["claim_results"][0]["claim"] == "The Earth is round."
    assert result["claim_results"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
