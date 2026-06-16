"""Configuration for claim extractor nodes."""

from __future__ import annotations

# Ollama context window for extractor stages (capped by OLLAMA_NUM_CTX in .env).
EXTRACTOR_NUM_CTX: int = 8192

SELECTION_CONFIG = {
    "completions": 3,
    "min_successes": 2,
    "temperature": 0.2,
    "num_predict": 512,
    "num_ctx": EXTRACTOR_NUM_CTX,
}
DISAMBIGUATION_CONFIG = {
    "completions": 3,
    "min_successes": 2,
    "temperature": 0.2,
    "num_predict": 768,
    "num_ctx": EXTRACTOR_NUM_CTX,
}
DECOMPOSITION_CONFIG = {
    "completions": 1,
    "min_successes": 1,
    "temperature": 0.0,
    "num_predict": 1024,
    "num_ctx": EXTRACTOR_NUM_CTX,
}
FIDELITY_CONFIG = {
    "temperature": 0.0,
    "num_predict": 256,
    "num_ctx": EXTRACTOR_NUM_CTX,
}
VALIDATION_CONFIG = {
    "temperature": 0.0,
    "num_predict": 256,
    "num_ctx": EXTRACTOR_NUM_CTX,
}

# Per-stage context windows.
CONTEXT_WINDOWS = {
    "selection": {"preceding_sentences": 5, "following_sentences": 5},
    # Shared by disambiguation and decomposition (preceding-only context).
    "preceding_only": {"preceding_sentences": 5, "following_sentences": 0},
}
