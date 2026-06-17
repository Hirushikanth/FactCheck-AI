from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from factcheck.http.pinned_fetch import fetch_html_pinned
from factcheck.http.url_policy import UrlPolicyError


async def test_fetch_html_pinned_returns_html_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body>Hello</body></html>",
        )

    with patch(
        "factcheck.http.pinned_fetch.resolve_public_ip",
        return_value="93.184.216.34",
    ):
        html = await fetch_html_pinned(
            "https://example.com/page",
            transport=httpx.MockTransport(handler),
        )

    assert "Hello" in html


async def test_fetch_html_pinned_blocks_redirect_to_internal_target() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="should not reach")

    with patch(
        "factcheck.http.pinned_fetch.resolve_public_ip",
        return_value="93.184.216.34",
    ):
        with pytest.raises(UrlPolicyError):
            await fetch_html_pinned(
                "https://example.com/start",
                transport=httpx.MockTransport(handler),
            )

    assert calls == 1
