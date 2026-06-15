from __future__ import annotations

from factcheck.search import SearchHit
from factcheck.verifier.schemas import EvidenceItem
from factcheck.verifier.utils import format_evidence, heuristic_prefilter_hits
from factcheck.verifier.utils.credibility import classify_domain, credibility_tier_label


def test_classify_gov_edu_as_high() -> None:
    assert classify_domain("https://www.cdc.gov/flu/index.html") == "high"
    assert classify_domain("https://www.ox.ac.uk/about") == "high"


def test_classify_reddit_as_low() -> None:
    assert classify_domain("https://www.reddit.com/r/news/comments/abc") == "low"


def test_classify_wikipedia_as_medium() -> None:
    assert classify_domain("https://en.wikipedia.org/wiki/Earth") == "medium"


def test_credibility_boost_ranks_gov_above_reddit() -> None:
    shared_snippet = "Vaccines reduce disease transmission according to public health guidance."
    gov_hit = SearchHit(
        url="https://www.cdc.gov/vaccines/overview",
        title="Vaccine overview",
        snippet=shared_snippet,
    )
    reddit_hit = SearchHit(
        url="https://www.reddit.com/r/health/comments/example",
        title="Health discussion",
        snippet=shared_snippet,
    )
    claim = "Vaccines reduce disease transmission"

    ranked = heuristic_prefilter_hits(claim, [reddit_hit, gov_hit], top_n=2)

    assert ranked[0][0].url == "https://www.cdc.gov/vaccines/overview"


def test_format_evidence_includes_source_tier() -> None:
    formatted = format_evidence(
        [
            EvidenceItem(
                url="https://www.cdc.gov/example",
                title="CDC page",
                snippet="Public health guidance.",
                credibility_tier="high",
            )
        ]
    )

    assert f"Source tier: {credibility_tier_label('high')}" in formatted
