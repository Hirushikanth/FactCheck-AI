"""Tests for dialogue audit fixes (intent, tokens, errors, prefixes, caching)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from factcheck.dialogue.nodes.forward_to_pipeline import forward_to_pipeline_node
from factcheck.dialogue.schemas import DialogueState


def _base_state(**overrides) -> DialogueState:
    state: DialogueState = {
        "session_id": "sess-1",
        "original_text": "Original article text.",
        "claim_results": [],
        "final_report": "# Report",
        "_compressed_fc_context": "=== FACT-CHECK RESULTS ===",
        "dialogue_history": [],
        "conversation_summary": None,
        "current_user_message": "hello",
        "classified_intent": "clarification",
        "rewritten_query": None,
        "dialogue_response": None,
        "_assembled_messages": None,
        "estimated_context_tokens": 0,
        "needs_compression": False,
        "needs_new_factcheck": False,
        "new_claim_text": None,
        "error_message": None,
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Fact-check this: The moon is cheese.", "The moon is cheese."),
        ("check this: Solar is 40% of US electricity", "Solar is 40% of US electricity"),
        ("Can you also check: Ban plastics by 2025", "Ban plastics by 2025"),
    ],
)
async def test_forward_to_pipeline_strips_prefixes(message: str, expected: str) -> None:
    result = await forward_to_pipeline_node(
        _base_state(current_user_message=message, classified_intent="new_claim")
    )
    assert result["new_claim_text"] == expected
    assert result["needs_new_factcheck"] is True


async def test_compress_history_failure_does_not_set_error_message(monkeypatch) -> None:
    import factcheck.dialogue.nodes.compress_history as compress_module

    failing_llm = AsyncMock()
    failing_llm.ainvoke = AsyncMock(side_effect=RuntimeError("compressor down"))
    monkeypatch.setattr(
        compress_module,
        "get_dialogue_compressor_llm",
        lambda **kw: failing_llm,
    )

    from factcheck.dialogue.nodes.compress_history import compress_history_node

    history = [
        {
            "role": "user",
            "content": f"Message {i}",
            "timestamp": float(i),
            "intent": None,
            "token_estimate": 10,
        }
        for i in range(8)
    ]
    result = await compress_history_node(
        _base_state(dialogue_history=history, needs_compression=True)
    )

    assert result["needs_compression"] is False
    assert "error_message" not in result


async def test_generate_response_clears_prior_error_message(monkeypatch) -> None:
    import factcheck.dialogue.nodes.generate_response as gen_module

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Grounded answer."))
    monkeypatch.setattr(gen_module, "get_dialogue_llm", lambda **kw: llm)

    from factcheck.dialogue.nodes.generate_response import generate_response_node

    result = await generate_response_node(
        _base_state(
            error_message="old compressor error",
            _assembled_messages=[{"role": "user", "content": "Why?"}],
        )
    )

    assert result["dialogue_response"] == "Grounded answer."
    assert result["error_message"] is None


def _empty_length_response(eval_count: int = 512) -> MagicMock:
    msg = MagicMock()
    msg.content = ""
    msg.response_metadata = {"done_reason": "length", "eval_count": eval_count}
    return msg


async def test_generate_response_retries_on_empty_length(monkeypatch) -> None:
    import factcheck.dialogue.config as dialogue_config
    import factcheck.dialogue.nodes.generate_response as gen_module

    retry_llm = AsyncMock()
    retry_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Retry succeeded."))

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_empty_length_response())
    llm.bind = MagicMock(return_value=retry_llm)
    monkeypatch.setattr(gen_module, "get_dialogue_llm", lambda **kw: llm)

    from factcheck.dialogue.nodes.generate_response import generate_response_node

    result = await generate_response_node(
        _base_state(_assembled_messages=[{"role": "user", "content": "Why?"}])
    )

    assert result["dialogue_response"] == "Retry succeeded."
    assert llm.ainvoke.await_count == 1
    llm.bind.assert_called_once_with(num_predict=dialogue_config.DIALOGUE_NUM_PREDICT_RETRY)
    retry_llm.ainvoke.assert_awaited_once()


async def test_generate_response_fallback_when_retry_also_empty(monkeypatch, caplog) -> None:
    import factcheck.dialogue.nodes.generate_response as gen_module

    retry_llm = AsyncMock()
    retry_llm.ainvoke = AsyncMock(return_value=_empty_length_response(eval_count=4096))

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_empty_length_response())
    llm.bind = MagicMock(return_value=retry_llm)
    monkeypatch.setattr(gen_module, "get_dialogue_llm", lambda **kw: llm)

    from factcheck.dialogue.nodes.generate_response import (
        _FALLBACK_RESPONSE,
        generate_response_node,
    )

    with caplog.at_level("WARNING"):
        result = await generate_response_node(
            _base_state(_assembled_messages=[{"role": "user", "content": "Why?"}])
        )

    assert result["dialogue_response"] == _FALLBACK_RESPONSE
    assert any("Using fallback" in record.message for record in caplog.records)


async def test_generate_response_no_retry_when_content_present(monkeypatch) -> None:
    import factcheck.dialogue.nodes.generate_response as gen_module

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Direct answer."))
    llm.bind = MagicMock()
    monkeypatch.setattr(gen_module, "get_dialogue_llm", lambda **kw: llm)

    from factcheck.dialogue.nodes.generate_response import generate_response_node

    result = await generate_response_node(
        _base_state(_assembled_messages=[{"role": "user", "content": "Why?"}])
    )

    assert result["dialogue_response"] == "Direct answer."
    llm.bind.assert_not_called()


async def test_estimate_tokens_uses_windowed_history_not_full_history() -> None:
    from factcheck.dialogue.nodes.estimate_tokens import _estimate_prompt_tokens

    small_turn = {
        "role": "assistant",
        "content": "short reply",
        "timestamp": 2.0,
        "intent": None,
        "token_estimate": 5,
    }
    fat_turn = {
        "role": "user",
        "content": "x" * 5000,
        "timestamp": 1.0,
        "intent": None,
        "token_estimate": 2000,
    }

    short_history_state = _base_state(dialogue_history=[small_turn, small_turn])
    long_history_state = _base_state(
        dialogue_history=[fat_turn] * 20 + [small_turn, small_turn]
    )

    short_total = _estimate_prompt_tokens(short_history_state)
    long_total = _estimate_prompt_tokens(long_history_state)

    # Windowing should keep the estimate stable despite many old fat turns.
    assert abs(long_total - short_total) < 50


async def test_init_context_skips_when_cache_present() -> None:
    from factcheck.dialogue.nodes.init_context import init_context_node

    cached = "=== cached context ==="
    result = await init_context_node(
        _base_state(
            _compressed_fc_context=cached,
            _fc_context_covers_sequence=1,
            _latest_run_sequence=1,
            claim_results=[{"claim": "A"}],
        )
    )
    assert result == {}


async def test_init_context_rebuilds_when_cache_stale() -> None:
    from factcheck.dialogue.nodes.init_context import init_context_node

    result = await init_context_node(
        _base_state(
            _compressed_fc_context="=== stale cache ===",
            _fc_context_covers_sequence=1,
            _latest_run_sequence=2,
            fact_check_runs=[
                {
                    "sequence": 1,
                    "raw_input": "First.",
                    "claim_results": [{"claim": "A", "verdict": "SUPPORTED"}],
                },
                {
                    "sequence": 2,
                    "raw_input": "Second.",
                    "claim_results": [{"claim": "B", "verdict": "REFUTED"}],
                },
            ],
        )
    )
    assert result.get("_compressed_fc_context")
    assert "FACT-CHECK RUN 1" in result["_compressed_fc_context"]
    assert "FACT-CHECK RUN 2" in result["_compressed_fc_context"]
    assert result.get("_fc_context_covers_sequence") == 2


async def test_run_dialogue_returns_classified_intent(monkeypatch) -> None:
    import factcheck.dialogue.nodes.acknowledge_new_claim as _ack
    import factcheck.dialogue.nodes.classify_intent as _ci
    import factcheck.dialogue.nodes.compress_history as _ch
    import factcheck.dialogue.nodes.generate_response as _gr
    import factcheck.dialogue.nodes.rewrite_query as _rq

    def _llm(text: str) -> MagicMock:
        msg = MagicMock()
        msg.content = text
        return msg

    classifier = AsyncMock()
    classifier.ainvoke = AsyncMock(return_value=_llm("out_of_scope"))
    monkeypatch.setattr(_ci, "get_dialogue_classifier_llm", lambda **kw: classifier)

    rewriter = AsyncMock()
    rewriter.ainvoke = AsyncMock(return_value=_llm("rewritten"))
    monkeypatch.setattr(_rq, "get_dialogue_rewriter_llm", lambda **kw: rewriter)

    generator = AsyncMock()
    generator.ainvoke = AsyncMock(return_value=_llm("Declined politely."))
    monkeypatch.setattr(_gr, "get_dialogue_llm", lambda **kw: generator)

    compressor = AsyncMock()
    compressor.ainvoke = AsyncMock(return_value=_llm("summary"))
    monkeypatch.setattr(_ch, "get_dialogue_compressor_llm", lambda **kw: compressor)

    ack = AsyncMock()
    ack.ainvoke = AsyncMock(return_value=_llm("queued"))
    monkeypatch.setattr(_ack, "get_dialogue_acknowledge_llm", lambda **kw: ack)

    from factcheck.dialogue import run_dialogue

    result = await run_dialogue(
        session_id="intent-test",
        user_message="Write me a poem",
        raw_input="The sky is blue.",
        claim_results=[
            {
                "claim": "The sky is blue.",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence": ["Evidence"],
                "sources": ["https://example.com"],
                "reasoning": "Supported.",
                "search_queries": ["sky blue"],
            }
        ],
    )

    assert result["intent"] == "out_of_scope"


async def test_run_dialogue_threads_compressed_fc_context(monkeypatch) -> None:
    import factcheck.dialogue.nodes.acknowledge_new_claim as _ack
    import factcheck.dialogue.nodes.classify_intent as _ci
    import factcheck.dialogue.nodes.compress_history as _ch
    import factcheck.dialogue.nodes.generate_response as _gr
    import factcheck.dialogue.nodes.rewrite_query as _rq

    def _llm(text: str) -> MagicMock:
        msg = MagicMock()
        msg.content = text
        return msg

    for module, factory_name, response in [
        (_ci, "get_dialogue_classifier_llm", "clarification"),
        (_rq, "get_dialogue_rewriter_llm", "rewritten"),
        (_gr, "get_dialogue_llm", "Answer."),
        (_ch, "get_dialogue_compressor_llm", "summary"),
        (_ack, "get_dialogue_acknowledge_llm", "queued"),
    ]:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_llm(response))

        def _factory(_llm=llm, **kwargs):
            return _llm

        monkeypatch.setattr(module, factory_name, _factory)

    from factcheck.dialogue import run_dialogue

    result1 = await run_dialogue(
        session_id="cache-test",
        user_message="First question",
        raw_input="Text",
        claim_results=[],
        compressed_fc_context=None,
    )
    assert result1["compressed_fc_context"] is not None

    result2 = await run_dialogue(
        session_id="cache-test",
        user_message="Second question",
        raw_input="Text",
        claim_results=[],
        compressed_fc_context=result1["compressed_fc_context"],
    )

    assert result2["compressed_fc_context"] == result1["compressed_fc_context"]
