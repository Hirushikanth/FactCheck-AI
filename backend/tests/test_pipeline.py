from factcheck.graph.pipeline import build_graph


def test_stub_pipeline_compiles_and_updates_current_agent() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "raw_input": "The Earth is round.",
            "extracted_claims": ["The Earth is round."],
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
    assert len(result["claim_results"]) == 1
    assert result["claim_results"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
