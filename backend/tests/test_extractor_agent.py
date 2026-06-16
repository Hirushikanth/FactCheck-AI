from __future__ import annotations

from factcheck.agents import extractor
from factcheck.agents.extractor import extractor_node
from factcheck.extractor import ExtractorRunResult
from factcheck.extractor.schemas import ExtractorStageFailure, ValidatedClaim


def _state(raw_input: str):
    return {
        "raw_input": raw_input,
        "extracted_claims": [],
        "claim_results": [],
        "final_report": None,
        "messages": [],
        "current_agent": "",
        "session_id": "test-session",
        "error": None,
        "status": "idle",
    }


async def test_extractor_node_writes_ordered_case_insensitive_unique_claims(monkeypatch) -> None:
    first_claim = ValidatedClaim(
        claim_text="Ada Lovelace wrote the first algorithm.",
        is_complete_declarative=True,
        disambiguated_sentence="Ada Lovelace wrote the first algorithm.",
        original_sentence="Ada wrote the first algorithm.",
        original_index=0,
    )
    duplicate_claim = ValidatedClaim(
        claim_text="  ada lovelace wrote the first algorithm. ",
        is_complete_declarative=True,
        disambiguated_sentence="Ada Lovelace wrote the first algorithm.",
        original_sentence="Ada wrote the first algorithm.",
        original_index=0,
    )
    second_claim = ValidatedClaim(
        claim_text="Charles Babbage designed the Analytical Engine.",
        is_complete_declarative=True,
        disambiguated_sentence="Charles Babbage designed the Analytical Engine.",
        original_sentence="Charles Babbage designed the Analytical Engine.",
        original_index=1,
    )

    async def fake_run_extractor(raw_input: str) -> ExtractorRunResult:
        assert raw_input == "Ada wrote the first algorithm."
        return ExtractorRunResult(
            claims=[first_claim, duplicate_claim, second_claim],
            stage_failures=[],
        )

    monkeypatch.setattr(extractor, "run_extractor", fake_run_extractor)

    result = await extractor_node(_state("Ada wrote the first algorithm."))

    assert result == {
        "current_agent": "extractor",
        "extracted_claims": [
            first_claim,
            second_claim,
        ],
    }


async def test_extractor_node_emits_stage_failed_sse(monkeypatch) -> None:
    pushed: list[tuple[str, str, dict]] = []

    async def capture_push(session_id: str, event: str, data: dict) -> None:
        pushed.append((session_id, event, data))

    failure = ExtractorStageFailure(
        stage="selection",
        sentence="the earth is flat",
        reason="voting_failed",
        successes=1,
        attempts=3,
    )

    async def fake_run_extractor(raw_input: str) -> ExtractorRunResult:
        return ExtractorRunResult(claims=[], stage_failures=[failure])

    monkeypatch.setattr(extractor, "run_extractor", fake_run_extractor)
    monkeypatch.setattr(extractor, "push_event", capture_push)

    await extractor_node(_state("the earth is flat"))

    assert len(pushed) == 1
    session_id, event_name, payload = pushed[0]
    assert session_id == "test-session"
    assert event_name == "extractor_stage_failed"
    assert payload["stage"] == "selection"
    assert payload["sentence"] == "the earth is flat"
    assert payload["reason"] == "voting_failed"
    assert payload["successes"] == 1
    assert payload["attempts"] == 3
    assert "timestamp" in payload
