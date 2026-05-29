"""Text helpers for extractor context handling."""

from __future__ import annotations


def remove_following_sentences(context_for_llm: str) -> str:
    """Remove the following-sentences section from a context window."""

    return context_for_llm.split("\n[Following Sentences:]", maxsplit=1)[0]
