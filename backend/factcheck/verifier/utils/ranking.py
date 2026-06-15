"""BM25 ranking for verifier search-hit pre-filtering."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from factcheck.search import SearchHit
from factcheck.verifier.config import BM25_B, BM25_K1


TokenizeFn = Callable[[str], Counter[str]]


@dataclass(frozen=True)
class Bm25Corpus:
    """Corpus statistics for in-set Okapi BM25 scoring."""

    doc_freq: dict[str, int]
    avg_doc_length: float
    num_docs: int


def _hit_text(hit: SearchHit) -> str:
    return f"{hit.title} {hit.snippet}"


def _idf(term: str, *, doc_freq: dict[str, int], num_docs: int) -> float:
    df = doc_freq.get(term, 0)
    return math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)


def build_bm25_corpus(hits: list[SearchHit], *, tokenize: TokenizeFn) -> Bm25Corpus:
    """Build document-frequency statistics from a search result set."""
    if not hits:
        return Bm25Corpus(doc_freq={}, avg_doc_length=0.0, num_docs=0)

    doc_freq: dict[str, int] = {}
    total_length = 0
    for hit in hits:
        term_counts = tokenize(_hit_text(hit))
        total_length += sum(term_counts.values())
        for term in term_counts:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_doc_length = total_length / len(hits) if hits else 0.0
    return Bm25Corpus(doc_freq=doc_freq, avg_doc_length=avg_doc_length, num_docs=len(hits))


def bm25_score(
    claim: str,
    hit: SearchHit,
    corpus: Bm25Corpus,
    *,
    tokenize: TokenizeFn,
    query_tokenize: Callable[[str], set[str]],
) -> float:
    """Score a hit against a claim using Okapi BM25 over the result-set corpus."""
    if corpus.num_docs == 0 or corpus.avg_doc_length == 0.0:
        return 0.0

    query_terms = query_tokenize(claim)
    if not query_terms:
        return 0.0

    doc_term_counts = tokenize(_hit_text(hit))
    if not doc_term_counts:
        return 0.0

    doc_length = sum(doc_term_counts.values())
    length_norm = 1.0 - BM25_B + BM25_B * (doc_length / corpus.avg_doc_length)

    score = 0.0
    for term in query_terms:
        term_freq = doc_term_counts.get(term, 0)
        if term_freq == 0:
            continue

        idf = _idf(term, doc_freq=corpus.doc_freq, num_docs=corpus.num_docs)
        tf_norm = (term_freq * (BM25_K1 + 1.0)) / (term_freq + BM25_K1 * length_norm)
        score += idf * tf_norm

    return score
