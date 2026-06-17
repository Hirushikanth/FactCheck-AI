"""Shared search result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    """A normalized search result used by future verifier code."""

    url: str
    title: str = ""
    snippet: str = ""
    page_text: str | None = None
