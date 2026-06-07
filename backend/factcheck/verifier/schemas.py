"""Internal data models for the verifier subgraph."""

from __future__ import annotations

from pydantic import BaseModel, Field

from factcheck.search.models import SearchHit
from factcheck.state import ClaimResult


class EvidenceItem(BaseModel):
    """A search hit selected as relevant evidence for a claim."""

    url: str
    title: str = ""
    snippet: str
    relevance_score: float


class VerifierState(BaseModel):
    """State object used inside the verifier subgraph for one claim."""

    claim: str
    search_queries: list[str] = Field(default_factory=list)
    raw_hits: list[SearchHit] = Field(default_factory=list)
    ranked_evidence: list[EvidenceItem] = Field(default_factory=list)
    claim_result: ClaimResult | None = None
