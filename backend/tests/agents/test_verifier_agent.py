"""Tests for the parallel verifier agent node."""

from __future__ import annotations

import logging

import pytest

from factcheck.agents import verifier as verifier_agent
from factcheck.extractor.schemas import ValidatedClaim


def _validated_claim(claim_text: str, *, original_index: int = 0) -> ValidatedClaim:
    return ValidatedClaim(
        claim_text=claim_text,
        is_complete_declarative=True,
        disambiguated_sentence=claim_text,
        original_sentence=claim_text,
        original_index=original_index,
    )


@pytest.mark.asyncio
async def test_verifier_node_emits_verdict_ready_for_each_claim(monkeypatch) -> None:
    claims = [
        _validated_claim("The Earth is round.", original_index=0),
        _validated_claim("Water boils at 100C.", original_index=1),
    ]
    pushed_events: list[dict] = []

    async def fake_run_verifier(claim: ValidatedClaim):
        return {
            "claim": claim.claim_text,
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "evidence": ["Evidence."],
            "sources": ["https://example.com"],
            "reasoning": "Supported.",
            "search_queries": ["query"],
        }

    async def capture_push(session_id: str, event: str, data: dict) -> None:
        pushed_events.append({"event": event, "data": data, "session_id": session_id})

    monkeypatch.setattr(verifier_agent, "run_verifier", fake_run_verifier)
    monkeypatch.setattr(verifier_agent, "push_event", capture_push)

    result = await verifier_agent.verifier_node(
        {
            "raw_input": "Compound input.",
            "extracted_claims": claims,
            "claim_results": [],
            "final_report": None,
            "messages": [],
            "current_agent": "",
            "session_id": "sess-verifier",
            "error": None,
            "status": "idle",
        }
    )

    verdict_events = [event for event in pushed_events if event["event"] == "verdict_ready"]
    assert len(verdict_events) == 2
    assert [event["data"]["index"] for event in verdict_events] == [0, 1]
    assert all(event["data"]["total"] == 2 for event in verdict_events)
    assert len(result["claim_results"]) == 2


@pytest.mark.asyncio
async def test_verifier_node_counts_processing_errors_not_reasoning_text(
    monkeypatch,
    caplog,
) -> None:
    claims = [
        _validated_claim("Claim that raises.", original_index=0),
        _validated_claim("Claim with failed wording.", original_index=1),
    ]

    async def fake_run_verifier(claim: ValidatedClaim):
        if claim.claim_text == "Claim that raises.":
            raise RuntimeError("boom")
        return {
            "claim": claim.claim_text,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.4,
            "evidence": ["Search failed to find peer-reviewed sources."],
            "sources": ["https://example.com"],
            "reasoning": "Searches failed to return authoritative sources.",
            "search_queries": ["query"],
        }

    async def noop_push(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(verifier_agent, "run_verifier", fake_run_verifier)
    monkeypatch.setattr(verifier_agent, "push_event", noop_push)

    with caplog.at_level(logging.WARNING):
        result = await verifier_agent.verifier_node(
            {
                "raw_input": "Compound input.",
                "extracted_claims": claims,
                "claim_results": [],
                "final_report": None,
                "messages": [],
                "current_agent": "",
                "session_id": "sess-verifier",
                "error": None,
                "status": "idle",
            }
        )

    assert result["claim_results"][0]["processing_status"] == "error"
    assert "processing_status" not in result["claim_results"][1]
    assert any("1/2 claims had verification errors" in record.message for record in caplog.records)
