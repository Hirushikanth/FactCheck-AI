"""Tests for token estimation utilities.

Covers:
  - estimate_tokens() basic correctness
  - get_windowed_history() respects turn count and token budget
  - truncate_to_tokens() hard-caps text
"""

from __future__ import annotations


from factcheck.dialogue.schemas import DialogueTurn
from factcheck.dialogue.utils.tokens import (
    estimate_tokens,
    estimate_turn_tokens,
    get_windowed_history,
    truncate_to_tokens,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _turn(role: str, content: str, token_estimate: int | None = None) -> DialogueTurn:
    est = token_estimate if token_estimate is not None else estimate_tokens(content)
    return DialogueTurn(
        role=role,
        content=content,
        timestamp=0.0,
        intent=None,
        token_estimate=est,
    )


def _turns(n: int, tokens_each: int = 50) -> list[DialogueTurn]:
    """Create *n* alternating user/assistant turns with ~*tokens_each* tokens each."""
    roles = ["user", "assistant"]
    return [
        _turn(roles[i % 2], " ".join(["word"] * tokens_each), token_estimate=tokens_each)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# estimate_tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_tokens_empty_string() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_nonempty_returns_positive() -> None:
    assert estimate_tokens("Hello, world!") > 0


def test_estimate_tokens_longer_text_more_tokens() -> None:
    short = estimate_tokens("Hello")
    long = estimate_tokens("Hello " * 100)
    assert long > short


# ─────────────────────────────────────────────────────────────────────────────
# estimate_turn_tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_turn_tokens_adds_overhead() -> None:
    turn = _turn("user", "Hello")
    base = estimate_tokens("Hello")
    assert estimate_turn_tokens(turn) == base + 4


# ─────────────────────────────────────────────────────────────────────────────
# get_windowed_history
# ─────────────────────────────────────────────────────────────────────────────

def test_windowed_history_empty_input() -> None:
    result = get_windowed_history([], max_turns=6, max_tokens=1800)
    assert result == []


def test_windowed_history_respects_max_turns() -> None:
    history = _turns(12, tokens_each=10)
    result = get_windowed_history(history, max_turns=6, max_tokens=9999)
    assert len(result) <= 6


def test_windowed_history_respects_token_budget() -> None:
    """Each turn has 50 tokens; budget of 120 allows at most 2 turns."""
    history = _turns(10, tokens_each=50)
    result = get_windowed_history(history, max_turns=10, max_tokens=120)
    total_tokens = sum(t["token_estimate"] for t in result)
    assert total_tokens <= 120


def test_windowed_history_returns_most_recent() -> None:
    """The window should contain the MOST RECENT turns."""
    history = _turns(10, tokens_each=10)
    # Mark the last 2 turns distinctively
    history[-2]["content"] = "RECENT_USER"
    history[-1]["content"] = "RECENT_ASSISTANT"

    result = get_windowed_history(history, max_turns=2, max_tokens=9999)
    assert len(result) == 2
    assert result[-1]["content"] == "RECENT_ASSISTANT"
    assert result[-2]["content"] == "RECENT_USER"


def test_windowed_history_preserves_chronological_order() -> None:
    """Returned list must be oldest-first (chronological order)."""
    history = [
        _turn("user", f"message {i}", token_estimate=5)
        for i in range(10)
    ]
    result = get_windowed_history(history, max_turns=4, max_tokens=9999)
    # Contents should be in the same order as history (oldest first in result)
    content_order = [t["content"] for t in result]
    assert content_order == sorted(content_order, key=lambda x: int(x.split()[-1]))


def test_windowed_history_fewer_turns_than_max() -> None:
    """If history has fewer turns than max_turns, all turns are returned."""
    history = _turns(3, tokens_each=10)
    result = get_windowed_history(history, max_turns=6, max_tokens=9999)
    assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# truncate_to_tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_truncate_to_tokens_short_text_unchanged() -> None:
    text = "Hello"
    assert truncate_to_tokens(text, max_tokens=200) == text


def test_truncate_to_tokens_truncates_long_text() -> None:
    long_text = "word " * 500  # ~500+ tokens
    result = truncate_to_tokens(long_text, max_tokens=50)
    assert "[truncated]" in result
    assert estimate_tokens(result) <= 60  # 50 + overhead for ' [truncated]'


def test_truncate_to_tokens_exact_limit() -> None:
    """Text that is exactly at the limit should not be modified."""
    text = "Hello world"
    limit = estimate_tokens(text)
    result = truncate_to_tokens(text, max_tokens=limit)
    assert result == text
