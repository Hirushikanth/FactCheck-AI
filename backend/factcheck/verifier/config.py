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

# Heuristic pre-filter: keep top N hits by BM25 before LLM evaluation
# (removes near-duplicate snippets and empty results before sending to evaluator)
RANKER_HEURISTIC_TOP_N = 10

# Okapi BM25 parameters for in-set search result re-ranking
BM25_K1 = 1.2
BM25_B = 0.75

# Domain-tier credibility adjustments during hit re-ranking
CREDIBILITY_HIGH_BOOST = 0.15
CREDIBILITY_MEDIUM_BOOST = 0.05
CREDIBILITY_LOW_PENALTY = 0.20

# LLM temperature settings per node
QUERY_GEN_TEMPERATURE = 0.0
EVALUATOR_TEMPERATURE = 0.0
