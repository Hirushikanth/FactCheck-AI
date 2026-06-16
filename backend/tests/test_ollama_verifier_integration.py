from __future__ import annotations

import os

import pytest

from factcheck.extractor.schemas import ValidatedClaim
from factcheck.state import Verdict
from factcheck.verifier import run_verifier


def _claim(text: str) -> ValidatedClaim:
    return ValidatedClaim(
        claim_text=text,
        is_complete_declarative=True,
        disambiguated_sentence=text,
        original_sentence=text,
        original_index=0,
    )


@pytest.mark.integration
async def test_ollama_verifier_supports_known_true_claim() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed verifier tests.")

    result = await run_verifier(_claim("The Earth orbits the Sun."))

    assert result["verdict"] == "SUPPORTED"
    assert result["claim"] == "The Earth orbits the Sun."
    assert result["search_queries"]


@pytest.mark.integration
async def test_ollama_verifier_handles_known_false_claim() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed verifier tests.")

    result = await run_verifier(_claim("The capital of France is Berlin."))

    allowed: tuple[Verdict, ...] = (
        "REFUTED",
        "CONFLICTING_EVIDENCE",
    )
    assert result["verdict"] in allowed
    assert result["claim"] == "The capital of France is Berlin."


@pytest.mark.integration
async def test_ollama_verifier_returns_verdict_for_obscure_claim() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed verifier tests.")

    result = await run_verifier(
        _claim("The Kandy Municipal Council approved a 12% water rate increase on 14 March 2024.")
    )

    assert result["verdict"] in (
        "SUPPORTED",
        "REFUTED",
        "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE",
    )
    assert result["claim"]


@pytest.mark.integration
async def test_ollama_verifier_refutes_vaccine_overload_claim() -> None:
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run Ollama-backed verifier tests.")

    result = await run_verifier(
        _claim("Vaccines overload your immune system if you take more than two in a year.")
    )

    assert result["verdict"] in ("REFUTED", "CONFLICTING_EVIDENCE")
    assert result["claim"]
