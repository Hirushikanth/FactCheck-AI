"""End-to-end tests for the run_dialogue() public entry point.

All LLM calls are monkeypatched — no live Ollama required.

Covers:
  - Happy path: returns DialogueOutput with correct fields
  - Error recovery: graph exception yields error in output, not a crash
  - new_claim: needs_new_factcheck=True and new_claim_text populated
  - History threading: subsequent turn includes prior turn in history
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _claim_result(
    claim: str = "The sky is blue.",
    verdict: str = "SUPPORTED",
    confidence: float = 0.95,
) -> dict:
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": ["Visible light scattering causes blue sky."],
        "sources": ["https://example.com/sky"],
        "reasoning": "Rayleigh scattering is well documented.",
        "search_queries": ["sky blue reason"],
    }


def _make_llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    return msg


def _patch_all_nodes(
    monkeypatch,
    *,
    intent: str = "clarification",
    response: str = "The sky was rated SUPPORTED.",
    ack: str = "New claim queued.",
):
    """Monkeypatch all dialogue LLM factories used in the graph nodes."""
    import factcheck.dialogue.nodes.classify_intent as _ci
    import factcheck.dialogue.nodes.rewrite_query as _rq
    import factcheck.dialogue.nodes.generate_response as _gr
    import factcheck.dialogue.nodes.compress_history as _ch
    import factcheck.dialogue.nodes.acknowledge_new_claim as _ack

    classifier = AsyncMock()
    classifier.ainvoke = AsyncMock(return_value=_make_llm_response(intent))
    monkeypatch.setattr(_ci, "get_dialogue_classifier_llm", lambda **kw: classifier)

    rewriter = AsyncMock()
    rewriter.ainvoke = AsyncMock(return_value=_make_llm_response("rewritten query"))
    monkeypatch.setattr(_rq, "get_dialogue_rewriter_llm", lambda **kw: rewriter)

    generator = AsyncMock()
    generator.ainvoke = AsyncMock(return_value=_make_llm_response(response))
    monkeypatch.setattr(_gr, "get_dialogue_llm", lambda **kw: generator)

    compressor = AsyncMock()
    compressor.ainvoke = AsyncMock(return_value=_make_llm_response("Old summary."))
    monkeypatch.setattr(_ch, "get_dialogue_compressor_llm", lambda **kw: compressor)

    ack_llm = AsyncMock()
    ack_llm.ainvoke = AsyncMock(return_value=_make_llm_response(ack))
    monkeypatch.setattr(_ack, "get_dialogue_acknowledge_llm", lambda **kw: ack_llm)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_run_dialogue_happy_path(monkeypatch) -> None:
    """run_dialogue() returns a well-formed DialogueOutput."""
    _patch_all_nodes(monkeypatch, response="The sky was rated SUPPORTED with 95% confidence.")

    from factcheck.dialogue import run_dialogue

    result = await run_dialogue(
        session_id="sess-001",
        user_message="What was the verdict for the sky claim?",
        raw_input="The sky is blue.",
        claim_results=[_claim_result()],
    )

    assert result["response"] == "The sky was rated SUPPORTED with 95% confidence."
    assert result["intent"] == "clarification"
    assert result["needs_new_factcheck"] is False
    assert result["new_claim_text"] is None
    assert result["error"] is None
    # History should have 2 turns
    assert len(result["dialogue_history"]) == 2


async def test_run_dialogue_new_claim_output(monkeypatch) -> None:
    """new_claim intent sets needs_new_factcheck=True and new_claim_text."""
    _patch_all_nodes(
        monkeypatch,
        intent="new_claim",
        ack="Your new claim has been queued.",
    )

    from factcheck.dialogue import run_dialogue

    result = await run_dialogue(
        session_id="sess-002",
        user_message="Check this: The moon is made of cheese.",
        raw_input="The sky is blue.",
        claim_results=[_claim_result()],
    )

    assert result["needs_new_factcheck"] is True
    assert result["new_claim_text"] is not None
    assert result["intent"] == "new_claim"
    assert "queued" in result["response"].lower()


async def test_run_dialogue_history_threading(monkeypatch) -> None:
    """Second call passes prior history; it is threaded into the new result."""
    _patch_all_nodes(monkeypatch, response="Claim 1 was SUPPORTED.")

    from factcheck.dialogue import run_dialogue

    # First turn
    result1 = await run_dialogue(
        session_id="sess-003",
        user_message="What was the verdict?",
        raw_input="The sky is blue.",
        claim_results=[_claim_result()],
        dialogue_history=[],
    )
    assert len(result1["dialogue_history"]) == 2

    # Second turn — pass the history from turn 1
    result2 = await run_dialogue(
        session_id="sess-003",
        user_message="How confident was it?",
        raw_input="The sky is blue.",
        claim_results=[_claim_result()],
        dialogue_history=result1["dialogue_history"],
    )
    # History should now have 4 turns
    assert len(result2["dialogue_history"]) == 4


async def test_run_dialogue_graph_error_returns_error_output(monkeypatch) -> None:
    """If the graph raises, run_dialogue() returns an error DialogueOutput, not a crash."""
    import factcheck.dialogue.graph as graph_module

    async def failing_ainvoke(state):
        raise RuntimeError("Ollama timeout")

    monkeypatch.setattr(graph_module.dialogue_graph, "ainvoke", failing_ainvoke)

    from factcheck.dialogue import run_dialogue

    result = await run_dialogue(
        session_id="sess-004",
        user_message="What happened?",
        raw_input="The sky is blue.",
        claim_results=[_claim_result()],
    )

    assert result["error"] is not None
    assert "Ollama timeout" in result["error"]
    assert result["response"] != ""  # fallback message is populated


async def test_run_dialogue_empty_claims(monkeypatch) -> None:
    """Empty claim_results produces a valid (not crashing) response."""
    _patch_all_nodes(monkeypatch, response="No claims were checked in this session.")

    from factcheck.dialogue import run_dialogue

    result = await run_dialogue(
        session_id="sess-005",
        user_message="What were the results?",
        raw_input="",
        claim_results=[],
    )

    assert result is not None
    assert isinstance(result["response"], str)
    assert result["error"] is None
