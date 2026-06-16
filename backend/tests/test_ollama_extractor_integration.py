from __future__ import annotations

import os

import pytest

from factcheck.extractor import run_extractor


@pytest.mark.integration
async def test_ollama_extractor_returns_claims_for_simple_text() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed extractor tests.")

    result = await run_extractor(
        "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
    )

    assert result.claims
    assert all(claim.claim_text.strip() for claim in result.claims)


@pytest.mark.integration
@pytest.mark.parametrize(
    "raw_input",
    [
        "the earth is flat",
        "The Earth is round.",
    ],
)
async def test_ollama_extractor_reliability_on_simple_factual_sentences(raw_input: str) -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed extractor tests.")

    successes = 0
    runs = 50
    for _ in range(runs):
        result = await run_extractor(raw_input)
        if result.claims:
            successes += 1

    assert successes >= 49, f"only {successes}/{runs} runs produced claims for {raw_input!r}"
