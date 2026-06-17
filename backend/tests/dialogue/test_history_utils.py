"""Tests for dialogue history helpers."""

from __future__ import annotations

from factcheck.dialogue.utils.history import append_turn_pair


def test_append_turn_pair_on_empty_history() -> None:
    user_turn = {
        "role": "user",
        "content": "Hello?",
        "timestamp": 1.0,
        "intent": "clarification",
        "token_estimate": 2,
    }
    assistant_turn = {
        "role": "assistant",
        "content": "Hi.",
        "timestamp": 2.0,
        "intent": None,
        "token_estimate": 1,
    }

    result = append_turn_pair([], user_turn, assistant_turn)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_append_turn_pair_replaces_trailing_user() -> None:
    optimistic_user = {
        "role": "user",
        "content": "Hello?",
        "timestamp": 1.0,
        "intent": None,
        "token_estimate": 0,
    }
    enriched_user = {
        "role": "user",
        "content": "Rewritten hello?",
        "timestamp": 99.0,
        "intent": "clarification",
        "token_estimate": 3,
    }
    assistant_turn = {
        "role": "assistant",
        "content": "Hi.",
        "timestamp": 2.0,
        "intent": None,
        "token_estimate": 1,
    }

    result = append_turn_pair([optimistic_user], enriched_user, assistant_turn)

    assert len(result) == 2
    assert result[0]["content"] == "Rewritten hello?"
    assert result[0]["intent"] == "clarification"
    assert result[0]["timestamp"] == 1.0
    assert result[1]["role"] == "assistant"
