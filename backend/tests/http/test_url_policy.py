from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from factcheck.http.url_policy import (
    UrlPolicyError,
    is_safe_citation_url,
    resolve_public_ip,
    validate_citation_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/",
        "https://169.254.169.254/latest/meta-data/",
        "https://10.0.0.1/",
        "https://[::1]/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://user@example.com/",
        "https://example.com:8080/",
        "https://localhost/",
    ],
)
def test_validate_citation_url_blocks_unsafe_urls(url: str) -> None:
    with pytest.raises(UrlPolicyError):
        validate_citation_url(url)
    assert is_safe_citation_url(url) is False


def test_validate_citation_url_allows_public_https_url() -> None:
    validate_citation_url("https://example.com/path")
    assert is_safe_citation_url("https://example.com/path") is True


async def test_resolve_public_ip_blocks_private_dns_results() -> None:
    def fake_getaddrinfo(host: str, port: int, *_args, **_kwargs):
        assert host == "evil.example"
        assert port == 443
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))]

    with patch("factcheck.http.url_policy.socket.getaddrinfo", fake_getaddrinfo):
        with pytest.raises(UrlPolicyError, match="No public IP"):
            await resolve_public_ip("evil.example", 443)


async def test_resolve_public_ip_returns_first_public_address() -> None:
    def fake_getaddrinfo(host: str, port: int, *_args, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    with patch("factcheck.http.url_policy.socket.getaddrinfo", fake_getaddrinfo):
        assert await resolve_public_ip("example.com", 443) == "93.184.216.34"
