from typing import get_args

from factcheck.state import ClaimResult, FactCheckState, PipelineStatus, Verdict
from factcheck.verifier import config as verifier_config
from factcheck.verifier.schemas import EvidenceItem, IntermediateAssessment, VerifierState


def test_pipeline_status_values_are_frozen() -> None:
    assert set(get_args(PipelineStatus)) == {"idle", "running", "done", "error"}


def test_verdict_values_are_frozen() -> None:
    assert set(get_args(Verdict)) == {
        "SUPPORTED",
        "REFUTED",
        "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE",
    }


def test_claim_result_fields_match_architecture_contract() -> None:
    assert set(ClaimResult.__annotations__) == {
        "claim",
        "verdict",
        "confidence",
        "evidence",
        "sources",
        "reasoning",
        "search_queries",
        "source_sentence",
        "fidelity_status",
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


def test_verifier_state_tracks_iterative_evidence_loop() -> None:
    state = VerifierState(claim_text="The Earth is round.")

    assert state.claim_text == "The Earth is round."
    assert state.source_sentence == "The Earth is round."
    assert state.current_query is None
    assert state.all_queries == []
    assert state.evidence == []
    assert state.iteration_count == 0
    assert state.max_iterations == verifier_config.MAX_ITERATIONS
    assert state.estimated_evidence_tokens == 0
    assert state.max_evidence_tokens == verifier_config.MAX_EVIDENCE_TOKENS


def test_verifier_schema_marks_influential_sources_and_missing_aspects() -> None:
    evidence = EvidenceItem(url="https://example.com", snippet="Evidence.", is_influential=True)
    assessment = IntermediateAssessment(
        needs_more_evidence=True,
        missing_aspects=["independent source"],
    )

    assert evidence.is_influential is True
    assert assessment.missing_aspects == ["independent source"]
