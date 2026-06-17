"""Helpers for dialogue history mutation."""

from __future__ import annotations

from factcheck.dialogue.schemas import DialogueTurn


def append_turn_pair(
    history: list[DialogueTurn],
    user_turn: DialogueTurn,
    assistant_turn: DialogueTurn,
) -> list[DialogueTurn]:
    """Append a user+assistant pair, replacing a trailing optimistic user row if present."""
    updated = list(history)
    if updated and updated[-1]["role"] == "user":
        user_turn = {**user_turn, "timestamp": updated[-1]["timestamp"]}
        updated[-1] = user_turn
        return updated + [assistant_turn]
    return updated + [user_turn, assistant_turn]
