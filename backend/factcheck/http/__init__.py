"""Safe HTTP utilities for outbound evidence fetching."""

from factcheck.http.pinned_fetch import fetch_html_pinned
from factcheck.http.url_policy import (
    UrlPolicyError,
    is_safe_citation_url,
    resolve_public_ip,
    validate_citation_url,
)

__all__ = [
    "UrlPolicyError",
    "fetch_html_pinned",
    "is_safe_citation_url",
    "resolve_public_ip",
    "validate_citation_url",
]
