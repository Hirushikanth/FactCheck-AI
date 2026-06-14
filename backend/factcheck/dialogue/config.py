"""Dialogue Agent configuration constants.

All token budgets and Ollama parameter values live here so that
node code contains zero magic numbers.  Values are tuned for
Mistral 7B running via Ollama on an RTX 3070 (8 GB VRAM).
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Ollama context window
# ─────────────────────────────────────────────────────────────────────────────

# Full KV-cache ceiling for Mistral 7B on 8 GB VRAM.  Must match
# OLLAMA_NUM_CTX in .env (or the Ollama Modelfile).
NUM_CTX: int = 8192

# ─────────────────────────────────────────────────────────────────────────────
# Token budget breakdown
# ─────────────────────────────────────────────────────────────────────────────

# Fixed overhead measured once (system prompt length).
# Re-measure if the system prompt in prompts.py changes.
SYSTEM_PROMPT_TOKENS: int = 380

# Maximum tokens allocated to the compressed fact-check context block.
MAX_FC_CONTEXT_TOKENS: int = 800

# Maximum tokens for the rolling conversation summary.
MAX_SUMMARY_TOKENS: int = 120

# Maximum tokens for the sliding-window recent history injected into the prompt.
SLIDING_WINDOW_TOKEN_BUDGET: int = 1800

# Maximum number of recent turns in the sliding window (user + assistant pairs).
SLIDING_WINDOW_MAX_TURNS: int = 6  # 3 pairs

# Hard cap on user message tokens before truncation.
MAX_USER_MESSAGE_TOKENS: int = 200

# Tokens reserved for the LLM response (and KV-cache headroom).
MAX_RESPONSE_TOKENS: int = 512

# Total prompt token ceiling before triggering history compression.
# Leaves MAX_RESPONSE_TOKENS + 3000 slack for per-turn LLM overhead.
COMPRESSION_THRESHOLD_TOKENS: int = NUM_CTX - 3500

# Compress when more than this many turns sit outside the sliding window.
COMPRESSION_THRESHOLD_TURNS: int = 4

# ─────────────────────────────────────────────────────────────────────────────
# Ollama generation parameters — per call type
# ─────────────────────────────────────────────────────────────────────────────

#: Main response generation
DIALOGUE_LLM_PARAMS: dict = {
    "num_ctx": NUM_CTX,
    "num_predict": MAX_RESPONSE_TOKENS,
    "temperature": 0.3,
    "top_p": 0.85,
    "repeat_penalty": 1.15,
    "stop": ["USER:", "Human:", "QUESTION:", "User:"],
}

#: Intent classification — fully deterministic, single-word output
CLASSIFIER_LLM_PARAMS: dict = {
    "num_ctx": NUM_CTX,
    "num_predict": 10,
    "temperature": 0.0,
    "stop": ["\n", "."],
}

#: Query rewriting — low temperature for accurate paraphrasing
REWRITER_LLM_PARAMS: dict = {
    "num_ctx": NUM_CTX,
    "num_predict": 80,
    "temperature": 0.1,
    "repeat_penalty": 1.1,
}

#: History compression — slightly warmer for natural summary prose
COMPRESSOR_LLM_PARAMS: dict = {
    "num_ctx": NUM_CTX,
    "num_predict": 120,
    "temperature": 0.2,
}

#: Acknowledgement for new claim submissions
ACKNOWLEDGE_LLM_PARAMS: dict = {
    "num_ctx": NUM_CTX,
    "num_predict": 80,
    "temperature": 0.3,
}
