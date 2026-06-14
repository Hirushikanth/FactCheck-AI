from __future__ import annotations

from factcheck.verifier.utils.framing import (
    extract_evaluation_frame,
    frame_tokens,
    snippet_looks_colloquial,
    snippet_matches_frame,
)


_BERRIES_CLAIM = (
    "Strawberries are not berries [according to botanical definitions of fruits]"
)


def test_extract_evaluation_frame_from_bracketed_claim() -> None:
    assert extract_evaluation_frame(_BERRIES_CLAIM) == (
        "according to botanical definitions of fruits"
    )


def test_snippet_looks_colloquial_detects_popular_usage() -> None:
    assert snippet_looks_colloquial("Strawberries are commonly called berries.")
    assert not snippet_looks_colloquial(
        "Botanically, strawberries are aggregate fruits, not true berries."
    )


def test_snippet_matches_botanical_frame() -> None:
    tokens = frame_tokens("according to botanical definitions of fruits")

    assert snippet_matches_frame(
        "Botanically, strawberries are aggregate fruits, not true berries.",
        tokens,
    )
    assert not snippet_matches_frame("Strawberries are commonly called berries.", tokens)
