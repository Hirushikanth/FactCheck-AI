"""Token estimation utilities for the Dialogue Agent.

Uses tiktoken (cl100k_base) as a proxy tokeniser for Mistral 7B.
Accuracy is within ±15 % for typical Latin-script text — sufficient
for the soft budget decisions made in this agent.

Nothing in this module makes LLM calls or performs I/O.
"""

from __future__ import annotations

import tiktoken

from factcheck.dialogue.schemas import DialogueTurn


# Encoder is loaded once at import time (thread-safe, immutable).
_encoder = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Return the estimated token count for *text*.

    Returns 0 for empty or non-string input rather than raising.
    """
    if not text:
        return 0
    return len(_encoder.encode(text))


def estimate_turn_tokens(turn: DialogueTurn) -> int:
    """Estimate tokens for a single dialogue turn.

    Adds 4 tokens of overhead per turn to account for the role header
    that a chat-formatted prompt inserts (e.g. ``<s>[INST] ... [/INST]``).
    """
    return estimate_tokens(turn.get("content", "")) + 4


def get_windowed_history(
    history: list[DialogueTurn],
    *,
    max_turns: int,
    max_tokens: int,
) -> list[DialogueTurn]:
    """Return the most recent turns that fit within *max_turns* and *max_tokens*.

    Walks the history backwards (most recent first) and accumulates turns
    until either the turn count or token budget is exhausted.  The returned
    list preserves chronological order (oldest first).

    Args:
        history:    The full dialogue history list.
        max_turns:  Hard limit on the number of turns returned.
        max_tokens: Soft token budget; a turn is excluded if adding it
                    would push the running total above this value.

    Returns:
        A sub-list of *history* in chronological order.
    """
    selected: list[DialogueTurn] = []
    token_count = 0

    for turn in reversed(history):
        turn_tokens = turn.get("token_estimate") or estimate_turn_tokens(turn)
        if token_count + turn_tokens > max_tokens:
            break
        if len(selected) >= max_turns:
            break
        selected.insert(0, turn)
        token_count += turn_tokens

    return selected


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate *text* to at most *max_tokens* tokens.

    Appends ``' [truncated]'`` if any content was removed.
    """
    encoded = _encoder.encode(text)
    if len(encoded) <= max_tokens:
        return text
    truncated = _encoder.decode(encoded[:max_tokens])
    return truncated + " [truncated]"
