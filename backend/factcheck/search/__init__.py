"""Search fallback API."""

from factcheck.search.fallback import search_with_fallback
from factcheck.search.models import SearchHit
from factcheck.search.providers import build_provider_chain

__all__ = ["SearchHit", "build_provider_chain", "search_with_fallback"]
