"""Tests for the Dialogue Agent StateGraph.

All LLM calls are monkeypatched so no live Ollama is required.

Covers:
  - Clarification path (first turn, no rewrite needed)
  - Clarification path with coreference → rewrite triggered
  - new_claim intent → forward_to_pipeline + acknowledge
  - Compression trigger (history exceeds token threshold)
  - Empty claim_results fallback (no crash, valid response)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock


from factcheck.dialogue.schemas import DialogueTurn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _claim_result(
    claim: str = "The Earth is round.",
    verdict: str = "SUPPORTED",
    confidence: float = 0.91,
) -> dict:
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": ["Multiple sources confirm this."],
        "sources": ["https://example.com/source"],
        "reasoning": "Well established scientific fact.",
        "search_queries": [f"{claim} evidence"],
    }


def _history_turn(role: str, content: str, intent: str | None = None) -> DialogueTurn:
    return DialogueTurn(
        role=role,
        content=content,
        timestamp=time.time(),
        intent=intent,
        token_estimate=len(content.split()),
    )


def _make_llm_response(text: str) -> MagicMock:
    """Return a mock that mimics a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = text
    return msg


def _patch_all_llm_factories(
    monkeypatch,
    *,
    intent: str = "clarification",
    rewritten: str = "What was the verdict for Claim 1?",
    response: str = "Claim 1 was rated SUPPORTED with 91% confidence.",
    summary: str = "User asked about Claim 1 verdict.",
    ack: str = "Your new claim has been queued for fact-checking.",
):
    """Monkeypatch all four dialogue LLM factories with async stubs."""
    import factcheck.dialogue.nodes.classify_intent as _ci
    import factcheck.dialogue.nodes.rewrite_query as _rq
    import factcheck.dialogue.nodes.generate_response as _gr
    import factcheck.dialogue.nodes.compress_history as _ch
    import factcheck.dialogue.nodes.acknowledge_new_claim as _ack

    # Intent classifier
    classifier_llm = AsyncMock()
    classifier_llm.ainvoke = AsyncMock(return_value=_make_llm_response(intent))
    monkeypatch.setattr(_ci, "get_dialogue_classifier_llm", lambda **kw: classifier_llm)

    # Rewriter
    rewriter_llm = AsyncMock()
    rewriter_llm.ainvoke = AsyncMock(return_value=_make_llm_response(rewritten))
    monkeypatch.setattr(_rq, "get_dialogue_rewriter_llm", lambda **kw: rewriter_llm)

    # Generator
    generator_llm = AsyncMock()
    generator_llm.ainvoke = AsyncMock(return_value=_make_llm_response(response))
    monkeypatch.setattr(_gr, "get_dialogue_llm", lambda **kw: generator_llm)

    # Compressor
    compressor_llm = AsyncMock()
    compressor_llm.ainvoke = AsyncMock(return_value=_make_llm_response(summary))
    monkeypatch.setattr(_ch, "get_dialogue_compressor_llm", lambda **kw: compressor_llm)

    # Acknowledge
    ack_llm = AsyncMock()
    ack_llm.ainvoke = AsyncMock(return_value=_make_llm_response(ack))
    monkeypatch.setattr(_ack, "get_dialogue_acknowledge_llm", lambda **kw: ack_llm)

    return {
        "classifier": classifier_llm,
        "rewriter": rewriter_llm,
        "generator": generator_llm,
        "compressor": compressor_llm,
        "ack": ack_llm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_clarification_path_first_turn(monkeypatch) -> None:
    """First turn: no history → rewrite is bypassed, response is generated."""
    mocks = _patch_all_llm_factories(monkeypatch, intent="clarification")

    from factcheck.dialogue.graph import build_dialogue_graph

    graph = build_dialogue_graph()
    from factcheck.dialogue.schemas import DialogueState

    state = DialogueState(
        session_id="test-session",
        original_text="The Earth is round.",
        claim_results=[_claim_result()],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=[],
        conversation_summary=None,
        current_user_message="What was the verdict for Claim 1?",
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    # Response was generated
    assert result["dialogue_response"] == "Claim 1 was rated SUPPORTED with 91% confidence."
    # History has 2 turns (user + assistant)
    assert len(result["dialogue_history"]) == 2
    assert result["dialogue_history"][0]["role"] == "user"
    assert result["dialogue_history"][1]["role"] == "assistant"
    # Rewriter was NOT called (first turn — no history)
    mocks["rewriter"].ainvoke.assert_not_called()
    # Not a new factcheck
    assert result["needs_new_factcheck"] is False


async def test_clarification_path_with_coreference(monkeypatch) -> None:
    """Second turn with coreference ('that') → rewrite is triggered."""
    mocks = _patch_all_llm_factories(
        monkeypatch,
        intent="clarification",
        rewritten="What sources were used to verify Claim 1?",
    )

    from factcheck.dialogue.graph import build_dialogue_graph
    from factcheck.dialogue.schemas import DialogueState

    graph = build_dialogue_graph()
    history = [
        _history_turn("user", "What was the verdict for Claim 1?", intent="clarification"),
        _history_turn("assistant", "Claim 1 was rated SUPPORTED with 91% confidence."),
    ]

    state = DialogueState(
        session_id="test-session",
        original_text="The Earth is round.",
        claim_results=[_claim_result()],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=history,
        conversation_summary=None,
        current_user_message="What sources did you use for that?",  # has 'that'
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    assert result["dialogue_response"] == "Claim 1 was rated SUPPORTED with 91% confidence."
    # Rewriter WAS called because 'that' triggers needs_rewriting()
    mocks["rewriter"].ainvoke.assert_called_once()
    assert len(result["dialogue_history"]) == 4  # 2 old + 2 new


async def test_new_claim_path(monkeypatch) -> None:
    """new_claim intent → forward_to_pipeline + acknowledge; no generate call."""
    mocks = _patch_all_llm_factories(
        monkeypatch,
        intent="new_claim",
        ack="Your new claim has been queued for fact-checking.",
    )

    from factcheck.dialogue.graph import build_dialogue_graph
    from factcheck.dialogue.schemas import DialogueState

    graph = build_dialogue_graph()

    state = DialogueState(
        session_id="test-session",
        original_text="The Earth is round.",
        claim_results=[_claim_result()],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=[],
        conversation_summary=None,
        current_user_message="Fact-check this: Solar panels now produce 50% of US power.",
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    # Pipeline handoff flags set
    assert result["needs_new_factcheck"] is True
    assert result["new_claim_text"] is not None
    assert result["classified_intent"] == "new_claim"
    # Acknowledgement response generated
    assert "queued" in result["dialogue_response"].lower()
    # Main generator NOT called (bypassed for new_claim path)
    mocks["generator"].ainvoke.assert_not_called()
    # Rewriter NOT called
    mocks["rewriter"].ainvoke.assert_not_called()


async def test_compression_triggered_when_over_threshold(monkeypatch) -> None:
    """When needs_compression is True, compress_history runs before classify_intent."""
    mocks = _patch_all_llm_factories(
        monkeypatch,
        intent="clarification",
        summary="User asked about Claim 1. Assistant confirmed it was SUPPORTED.",
    )

    from factcheck.dialogue.graph import build_dialogue_graph
    from factcheck.dialogue.schemas import DialogueState

    graph = build_dialogue_graph()

    # Build a history large enough that actual tiktoken count exceeds
    # COMPRESSION_THRESHOLD_TOKENS (= NUM_CTX - 3500 = 4692 tokens).
    # Each turn is ~250 tokens of real text; 20 turns = ~5000 tokens.
    fat_history = [
        DialogueTurn(
            role="user" if i % 2 == 0 else "assistant",
            content="word " * 250,  # ~250 real tokens per turn
            timestamp=float(i),
            intent="clarification" if i % 2 == 0 else None,
            token_estimate=250,
        )
        for i in range(20)
    ]

    state = DialogueState(
        session_id="test-session",
        original_text="Something was fact-checked.",
        claim_results=[_claim_result()],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=fat_history,
        conversation_summary=None,
        current_user_message="What was the main verdict?",
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    # Compressor was called
    mocks["compressor"].ainvoke.assert_called_once()
    # A new conversation summary was created
    assert result.get("conversation_summary") is not None
    assert "Claim 1" in result["conversation_summary"]["text"] or len(result["conversation_summary"]["text"]) > 0


async def test_empty_claim_results_no_crash(monkeypatch) -> None:
    """Graph completes without error even when no claims were checked."""
    _patch_all_llm_factories(monkeypatch, intent="clarification")

    from factcheck.dialogue.graph import build_dialogue_graph
    from factcheck.dialogue.schemas import DialogueState

    graph = build_dialogue_graph()

    state = DialogueState(
        session_id="empty-session",
        original_text="",
        claim_results=[],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=[],
        conversation_summary=None,
        current_user_message="What were the results?",
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    # Should complete without raising
    assert result is not None
    assert "dialogue_response" in result


async def test_out_of_scope_bypasses_rewrite(monkeypatch) -> None:
    """out_of_scope intent → rewrite is bypassed (no coreference resolution needed)."""
    mocks = _patch_all_llm_factories(monkeypatch, intent="out_of_scope")

    from factcheck.dialogue.graph import build_dialogue_graph
    from factcheck.dialogue.schemas import DialogueState

    graph = build_dialogue_graph()
    history = [_history_turn("user", "What is the verdict?", "clarification"),
               _history_turn("assistant", "It was SUPPORTED.")]

    state = DialogueState(
        session_id="test-session",
        original_text="Earth is round.",
        claim_results=[_claim_result()],
        final_report=None,
        _compressed_fc_context=None,
        dialogue_history=history,
        conversation_summary=None,
        # Message contains 'this' but intent is out_of_scope → rewrite bypassed
        current_user_message="Write me a poem about this topic.",
        classified_intent=None,
        rewritten_query=None,
        dialogue_response=None,
        _assembled_messages=None,
        estimated_context_tokens=0,
        needs_compression=False,
        needs_new_factcheck=False,
        new_claim_text=None,
        error_message=None,
    )

    result = await graph.ainvoke(state)

    # Rewriter NOT called for out_of_scope even though 'this' is in message
    mocks["rewriter"].ainvoke.assert_not_called()
    assert result["dialogue_response"] is not None
