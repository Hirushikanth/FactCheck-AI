"""Configuration for claim extractor nodes."""

from __future__ import annotations


SELECTION_CONFIG = {"completions": 3, "min_successes": 2, "temperature": 0.2}
DISAMBIGUATION_CONFIG = {"completions": 3, "min_successes": 2, "temperature": 0.2}
DECOMPOSITION_CONFIG = {"completions": 1, "min_successes": 1, "temperature": 0.0}
FIDELITY_CONFIG = {"temperature": 0.0}
VALIDATION_CONFIG = {"temperature": 0.0}

# Per-stage context windows.
CONTEXT_WINDOWS = {
    "selection": {"preceding_sentences": 5, "following_sentences": 5},
    # Shared by disambiguation and decomposition (preceding-only context).
    "preceding_only": {"preceding_sentences": 5, "following_sentences": 0},
}
