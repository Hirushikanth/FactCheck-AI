"""Citation URL validation and DNS resolution for safe outbound fetches."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

MAX_CITATION_URL_LENGTH = 2048

_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.goog",
})

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_PORTS = frozenset({80, 443})


class UrlPolicyError(ValueError):
    """Raised when a URL fails citation or fetch safety checks."""


def is_safe_citation_url(url: str) -> bool:
    """Return True when a URL is safe to store as evidence and attempt fetching."""

    try:
        validate_citation_url(url)
    except UrlPolicyError:
        return False
    return True


def validate_citation_url(url: str) -> None:
    """Validate URL shape for citation storage and outbound fetch."""

    if not url or not isinstance(url, str):
        raise UrlPolicyError("URL must be a non-empty string")

    trimmed = url.strip()
    if len(trimmed) > MAX_CITATION_URL_LENGTH:
        raise UrlPolicyError("URL exceeds maximum length")

    parts = urlsplit(trimmed)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UrlPolicyError(f"Unsupported URL scheme: {scheme!r}")

    if parts.username or parts.password:
        raise UrlPolicyError("URL userinfo is not allowed")

    host = parts.hostname
    if not host:
        raise UrlPolicyError("URL host is required")

    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in _ALLOWED_PORTS:
        raise UrlPolicyError(f"Unsupported URL port: {port}")

    normalized_host = _normalize_hostname(host)
    if normalized_host in _BLOCKED_HOSTNAMES:
        raise UrlPolicyError(f"Blocked hostname: {normalized_host}")

    if _is_literal_ip(normalized_host):
        _validate_ip_address(normalized_host)


def _normalize_hostname(host: str) -> str:
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise UrlPolicyError("Invalid hostname encoding") from exc


def _is_literal_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _validate_ip_address(host: str) -> None:
    address = ipaddress.ip_address(host)
    if _is_blocked_ip(address):
        raise UrlPolicyError(f"Blocked IP address: {host}")


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def resolve_public_ip(hostname: str, port: int) -> str:
    """Resolve hostname to the first public IP suitable for pinned fetch."""

    normalized_host = _normalize_hostname(hostname)
    if _is_literal_ip(normalized_host):
        _validate_ip_address(normalized_host)
        return normalized_host

    if normalized_host in _BLOCKED_HOSTNAMES:
        raise UrlPolicyError(f"Blocked hostname: {normalized_host}")

    try:
        addr_infos = await asyncio.to_thread(
            socket.getaddrinfo,
            normalized_host,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise UrlPolicyError(f"Failed to resolve hostname: {normalized_host}") from exc

    for addr_info in addr_infos:
        sockaddr = addr_info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            address = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not _is_blocked_ip(address):
            return ip_str

    raise UrlPolicyError(f"No public IP address found for hostname: {normalized_host}")
