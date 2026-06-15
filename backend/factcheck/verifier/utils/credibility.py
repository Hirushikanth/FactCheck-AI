"""Static domain-tier credibility heuristics for verifier evidence."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit


CredibilityTier = Literal["high", "medium", "low", "unknown"]

_HIGH_SUFFIXES = (".gov", ".edu", ".ac.uk", ".gov.uk")

_HIGH_DOMAINS = frozenset({
    "who.int",
    "nih.gov",
    "cdc.gov",
    "nature.com",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "snopes.com",
    "factcheck.org",
    "politifact.com",
})

_MEDIUM_DOMAINS = frozenset({
    "wikipedia.org",
    "britannica.com",
    "sciencedirect.com",
})

_LOW_DOMAINS = frozenset({
    "reddit.com",
    "quora.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
    "substack.com",
    "tiktok.com",
    "pinterest.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
})

_TIER_LABELS: dict[CredibilityTier, str] = {
    "high": "high-authority",
    "medium": "established reference",
    "low": "low-authority",
    "unknown": "unknown",
}


def _normalize_domain(url: str) -> str:
    return urlsplit(url).netloc.lower().lstrip("www.")


def _domain_matches(domain: str, candidate: str) -> bool:
    return domain == candidate or domain.endswith(f".{candidate}")


def _has_high_suffix(domain: str) -> bool:
    return any(domain.endswith(suffix) for suffix in _HIGH_SUFFIXES)


def classify_domain(url: str) -> CredibilityTier:
    """Classify a URL's domain into a static credibility tier."""
    domain = _normalize_domain(url)
    if not domain:
        return "unknown"

    if any(_domain_matches(domain, candidate) for candidate in _LOW_DOMAINS):
        return "low"

    if _has_high_suffix(domain) or any(
        _domain_matches(domain, candidate) for candidate in _HIGH_DOMAINS
    ):
        return "high"

    if any(_domain_matches(domain, candidate) for candidate in _MEDIUM_DOMAINS):
        return "medium"

    return "unknown"


def credibility_tier_label(tier: CredibilityTier) -> str:
    return _TIER_LABELS[tier]
