from __future__ import annotations

from factcheck.agents import extractor
from factcheck.agents.extractor import extractor_node


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
    async def fake_run_extractor(raw_input: str) -> list[str]:
        assert raw_input == "Ada wrote the first algorithm."
        return [
            "Ada Lovelace wrote the first algorithm.",
            "  ada lovelace wrote the first algorithm. ",
            "",
            "Charles Babbage designed the Analytical Engine.",
        ]

    monkeypatch.setattr(extractor, "run_extractor", fake_run_extractor)

    result = await extractor_node(_state("Ada wrote the first algorithm."))

    assert result == {
        "current_agent": "extractor",
        "extracted_claims": [
            "Ada Lovelace wrote the first algorithm.",
            "Charles Babbage designed the Analytical Engine.",
        ],
    }
