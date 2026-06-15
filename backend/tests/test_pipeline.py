from factcheck.graph.pipeline import build_graph
from factcheck.extractor.schemas import ValidatedClaim


def _validated_claim(
    claim_text: str,
    *,
    original_index: int = 0,
) -> ValidatedClaim:
    return ValidatedClaim(
        claim_text=claim_text,
        is_complete_declarative=True,
        disambiguated_sentence=claim_text,
        original_sentence=claim_text,
        original_index=original_index,
    )


async def test_pipeline_runs_extractor_before_verifier(monkeypatch) -> None:
    claim = _validated_claim("The Earth is round.")

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
    import factcheck.agents.reporter as reporter

    async def reporter_stub(state):
        assert state["claim_results"][0]["verdict"] == "SUPPORTED"
        return "# Fact-Check Report\n\n### Claim 1 - SUPPORTED"

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)
    monkeypatch.setattr(verifier, "run_verifier", verifier_stub)
    monkeypatch.setattr(reporter, "run_reporter", reporter_stub)

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
    assert result["final_report"].startswith("# Fact-Check Report")
    assert "SUPPORTED" in result["final_report"]
    assert result["extracted_claims"] == [claim]
    assert len(result["claim_results"]) == 1
    assert result["claim_results"][0]["verdict"] == "SUPPORTED"


async def test_pipeline_continues_after_single_claim_verifier_error(monkeypatch) -> None:
    claim = _validated_claim("The Earth is round.")

    async def extractor_stub(state):
        return {"current_agent": "extractor", "extracted_claims": [claim]}

    async def verifier_stub(verifier_claim: ValidatedClaim):
        assert verifier_claim == claim
        raise RuntimeError("boom")

    async def reporter_stub(state):
        return "# Fact-Check Report\n\n### Claim 1 - INSUFFICIENT_EVIDENCE"

    import factcheck.graph.pipeline as pipeline
    import factcheck.agents.verifier as verifier
    import factcheck.agents.reporter as reporter

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)
    monkeypatch.setattr(verifier, "run_verifier", verifier_stub)
    monkeypatch.setattr(reporter, "run_reporter", reporter_stub)

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
    assert result["error"] is None
    assert result["final_report"] is not None
    assert result["claim_results"][0]["claim"] == "The Earth is round."
    assert result["claim_results"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "failed" in result["claim_results"][0]["reasoning"].lower()


async def test_pipeline_verifies_all_claims_in_parallel(monkeypatch) -> None:
    claims = [
        _validated_claim("The Earth is round.", original_index=0),
        _validated_claim("Water boils at 100C.", original_index=1),
        _validated_claim("The Moon orbits Earth.", original_index=2),
    ]
    verified_claims: list[str] = []

    async def extractor_stub(state):
        return {"current_agent": "extractor", "extracted_claims": claims}

    async def verifier_stub(verifier_claim: ValidatedClaim):
        verified_claims.append(verifier_claim.claim_text)
        return {
            "claim": verifier_claim.claim_text,
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "evidence": ["Evidence."],
            "sources": ["https://example.com"],
            "reasoning": "Supported.",
            "search_queries": ["query"],
        }

    async def reporter_stub(state):
        assert len(state["claim_results"]) == 3
        return "# Fact-Check Report"

    import factcheck.graph.pipeline as pipeline
    import factcheck.agents.verifier as verifier
    import factcheck.agents.reporter as reporter

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)
    monkeypatch.setattr(verifier, "run_verifier", verifier_stub)
    monkeypatch.setattr(reporter, "run_reporter", reporter_stub)

    graph = build_graph()
    result = await graph.ainvoke(
        {
            "raw_input": "Compound input.",
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

    assert len(verified_claims) == 3
    assert {claim.claim_text for claim in claims} == set(verified_claims)
    assert len(result["claim_results"]) == 3
    assert all(result["verdict"] == "SUPPORTED" for result in result["claim_results"])
