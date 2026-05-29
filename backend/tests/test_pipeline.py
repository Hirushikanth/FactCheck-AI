from factcheck.graph.pipeline import build_graph


async def test_pipeline_runs_extractor_before_verifier(monkeypatch) -> None:
    async def extractor_stub(state):
        return {"current_agent": "extractor", "extracted_claims": ["The Earth is round."]}

    import factcheck.graph.pipeline as pipeline

    monkeypatch.setattr(pipeline, "extractor_node", extractor_stub)

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
    assert result["extracted_claims"] == ["The Earth is round."]
    assert len(result["claim_results"]) == 1
    assert result["claim_results"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
