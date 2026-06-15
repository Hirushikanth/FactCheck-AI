from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.verifier.utils import heuristic_prefilter_hits, token_counts, token_overlap_score, tokens
from factcheck.verifier.utils.ranking import bm25_score, build_bm25_corpus


def _generic_hit(index: int) -> SearchHit:
    return SearchHit(
        url=f"https://example.com/generic-{index}",
        title=f"Generic study {index}",
        snippet="A new study research data shows company results from analysis.",
    )


def test_bm25_prefers_rare_discriminative_terms() -> None:
    specific = SearchHit(
        url="https://nasa.gov/mars-rover",
        title="NASA Mars rover mission",
        snippet="NASA confirmed details of the Mars rover mission timeline.",
    )
    hits = [_generic_hit(1), _generic_hit(2), _generic_hit(3), specific]
    claim = "NASA Mars rover mission"

    ranked = heuristic_prefilter_hits(claim, hits, top_n=4)

    assert ranked[0][0].url == "https://nasa.gov/mars-rover"


def test_bm25_does_not_confuse_high_overlap_generic_hits() -> None:
    yellow_hit = SearchHit(
        url="https://science.example/sun-yellow",
        title="Sun color",
        snippet="Scientists confirm the sun appears yellow in the visible spectrum.",
    )
    bright_hit = SearchHit(
        url="https://science.example/sun-bright",
        title="Sun brightness",
        snippet="The sun is bright and emits intense light across study research data.",
    )
    claim = "The sun is yellow"

    ranked = heuristic_prefilter_hits(claim, [bright_hit, yellow_hit], top_n=2)

    assert ranked[0][0].url == "https://science.example/sun-yellow"
    assert bm25_score(claim, yellow_hit, build_bm25_corpus(
        [bright_hit, yellow_hit], tokenize=token_counts
    ), tokenize=token_counts, query_tokenize=tokens) > bm25_score(
        claim,
        bright_hit,
        build_bm25_corpus([bright_hit, yellow_hit], tokenize=token_counts),
        tokenize=token_counts,
        query_tokenize=tokens,
    )


def test_bm25_single_hit_falls_back_to_overlap() -> None:
    hit = SearchHit(
        url="https://example.com/earth",
        title="Earth",
        snippet="Earth is an oblate spheroid.",
    )
    claim = "Earth oblate spheroid"

    ranked = heuristic_prefilter_hits(claim, [hit], top_n=1)

    assert len(ranked) == 1
    assert ranked[0][1] == token_overlap_score(claim, hit)


def test_build_bm25_corpus_empty_returns_zero_score() -> None:
    corpus = build_bm25_corpus([], tokenize=token_counts)
    hit = SearchHit(url="https://example.com", title="T", snippet="text")

    assert corpus.num_docs == 0
    assert bm25_score("claim", hit, corpus, tokenize=token_counts, query_tokenize=tokens) == 0.0
