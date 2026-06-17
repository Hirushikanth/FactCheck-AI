"""Internal data models for the verifier subgraph."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from factcheck.search.models import SearchHit
from factcheck.state import ClaimResult, Verdict
from factcheck.verifier.config import MAX_EVIDENCE_TOKENS, MAX_ITERATIONS


ContentSource = Literal["fetched", "snippet"]
CredibilityTier = Literal["high", "medium", "low", "unknown"]


class EvidenceItem(BaseModel):
    """A search hit selected as relevant evidence for a claim."""

    url: str
    title: str = ""
    snippet: str
    content_source: ContentSource = "snippet"
    credibility_tier: CredibilityTier = "unknown"
    relevance_score: float = 0.0
    is_influential: bool = False


class IntermediateAssessment(BaseModel):
    """Evaluator assessment that decides whether another search round is useful."""

    needs_more_evidence: bool = False
    missing_aspects: list[str] = Field(default_factory=list)


class CachedEvaluation(BaseModel):
    """Last successful evaluator output, retained across iterations for fallback."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    core_predicate: str = ""
    predicate_resolved_by_evidence: bool = False
    refuting_sources: list[int] = Field(default_factory=list)
    needs_more_evidence: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
    influential_sources: list[int] = Field(default_factory=list)


class VerifierState(BaseModel):
    """State object used inside the verifier subgraph for one claim."""

    model_config = ConfigDict(populate_by_name=True)

    claim_text: str = Field(alias="claim")
    source_sentence: str | None = None
    disambiguated_sentence: str | None = None
    original_index: int | None = None
    fidelity_status: str | None = None
    current_query: str | None = None
    current_queries: list[str] = Field(default_factory=list)
    all_queries: list[str] = Field(default_factory=list)
    evidence: Annotated[list[EvidenceItem], add] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = MAX_ITERATIONS
    intermediate_assessment: IntermediateAssessment | None = None
    estimated_evidence_tokens: int = 0
    max_evidence_tokens: int = MAX_EVIDENCE_TOKENS
    raw_hits: list[SearchHit] = Field(default_factory=list)
    ranked_evidence: list[EvidenceItem] = Field(default_factory=list)
    claim_result: ClaimResult | None = None
    cached_evaluation: CachedEvaluation | None = None
    search_exhausted: bool = False

    @model_validator(mode="after")
    def _default_source_sentence(self) -> VerifierState:
        if self.source_sentence is None:
            self.source_sentence = self.claim_text
        return self

    @property
    def claim(self) -> str:
        """Compatibility alias while verifier nodes migrate to claim_text."""

        return self.claim_text

    @property
    def search_queries(self) -> list[str]:
        """Compatibility alias while verifier nodes migrate to all_queries."""

        return self.all_queries
