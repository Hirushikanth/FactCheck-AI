"""Pinned HTTP fetch with redirect re-validation for SSRF mitigation."""

from __future__ import annotations

import logging
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

from factcheck.http.url_policy import UrlPolicyError, resolve_public_ip, validate_citation_url


logger = logging.getLogger(__name__)

DEFAULT_FETCH_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_FETCH_BYTES = 512_000
DEFAULT_MAX_REDIRECTS = 3

_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

_USER_AGENT = (
    "Mozilla/5.0 (compatible; FactCheckBot/1.0; +https://academic-factchecker.example)"
)


async def fetch_html_pinned(
    url: str,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Fetch HTML using a DNS-pinned connection and validated redirect hops."""

    current_url = url.strip()
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT},
        transport=transport,
    ) as client:
        for hop in range(max_redirects + 1):
            validate_citation_url(current_url)
            parts = urlsplit(current_url)
            scheme = parts.scheme.lower()
            hostname = parts.hostname
            if not hostname:
                raise UrlPolicyError("URL host is required")

            port = parts.port or (443 if scheme == "https" else 80)
            pinned_ip = await resolve_public_ip(hostname, port)
            request_url = _build_pinned_url(scheme, pinned_ip, port, parts)

            headers = {"Host": hostname}
            extensions: dict[str, str] = {}
            if scheme == "https":
                extensions["sni_hostname"] = hostname

            response = await client.get(
                request_url,
                headers=headers,
                extensions=extensions,
            )

            if response.status_code in _REDIRECT_STATUS_CODES:
                location = response.headers.get("location")
                if not location:
                    raise UrlPolicyError("Redirect response missing Location header")
                if hop >= max_redirects:
                    raise UrlPolicyError("Too many redirects")
                current_url = urljoin(str(response.url), location)
                continue

            if response.status_code != 200:
                raise UrlPolicyError(f"Unexpected HTTP status: {response.status_code}")

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                raise UrlPolicyError(f"Non-HTML content-type: {content_type}")

            return await _read_limited_text(response, max_bytes)

    raise UrlPolicyError("Too many redirects")


def _build_pinned_url(
    scheme: str,
    pinned_ip: str,
    port: int,
    parts: SplitResult,
) -> str:
    host = pinned_ip
    if ":" in pinned_ip and not pinned_ip.startswith("["):
        host = f"[{pinned_ip}]"

    default_port = 443 if scheme == "https" else 80
    include_port = port != default_port
    netloc = f"{host}:{port}" if include_port else host
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


async def _read_limited_text(response: httpx.Response, max_bytes: int) -> str:
    content = response.content
    if len(content) > max_bytes:
        raise UrlPolicyError("Response body exceeds maximum size")
    return content.decode(response.encoding or "utf-8", errors="replace")
