"""Configuration for verifier nodes."""

from __future__ import annotations


# Iterative search loop limits
MAX_ITERATIONS = 5
QUERIES_PER_ITERATION = 2
MAX_SEARCH_QUERIES = MAX_ITERATIONS * QUERIES_PER_ITERATION

# Evidence collection limits
MAX_SNIPPET_WORDS = 150
MAX_EVIDENCE_TOKENS = 3072
FULL_PAGE_FETCH_TOP_N = 5

# LLM context window sizes per node
QUERY_GEN_NUM_CTX = 2048
EVAL_NUM_CTX = 6144

# Heuristic pre-filter: keep top N hits by token overlap before LLM evaluation
# (removes near-duplicate snippets and empty results before sending to evaluator)
RANKER_HEURISTIC_TOP_N = 10

# LLM temperature settings per node
QUERY_GEN_TEMPERATURE = 0.0
EVALUATOR_TEMPERATURE = 0.0
