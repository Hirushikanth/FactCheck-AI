from typing import get_args

from factcheck.state import ClaimResult, FactCheckState, PipelineStatus, Verdict


def test_pipeline_status_values_are_frozen() -> None:
    assert set(get_args(PipelineStatus)) == {"idle", "running", "done", "error"}


def test_verdict_values_are_frozen() -> None:
    assert set(get_args(Verdict)) == {"SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"}


def test_claim_result_fields_match_architecture_contract() -> None:
    assert set(ClaimResult.__annotations__) == {
        "claim",
        "verdict",
        "confidence",
        "evidence",
        "sources",
        "reasoning",
        "search_queries",
    }


def test_factcheck_state_fields_match_architecture_contract() -> None:
    assert set(FactCheckState.__annotations__) == {
        "raw_input",
        "extracted_claims",
        "claim_results",
        "final_report",
        "messages",
        "current_agent",
        "session_id",
        "error",
        "status",
    }
