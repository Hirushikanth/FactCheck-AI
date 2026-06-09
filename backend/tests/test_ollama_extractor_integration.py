from __future__ import annotations

import os

import pytest

from factcheck.extractor import run_extractor


@pytest.mark.integration
async def test_ollama_extractor_returns_claims_for_simple_text() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed extractor tests.")

    claims = await run_extractor(
        "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."
    )

    assert claims
    assert all(claim.claim_text.strip() for claim in claims)
