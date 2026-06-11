"""Internal data models for the reporter agent."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ReportVerdict(str, Enum):
    """Verdict values accepted from the verifier output."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


VERDICT_LABELS = {
    ReportVerdict.SUPPORTED: "Supported",
    ReportVerdict.REFUTED: "Refuted",
    ReportVerdict.INSUFFICIENT_EVIDENCE: "Insufficient evidence",
    ReportVerdict.CONFLICTING_EVIDENCE: "Conflicting evidence",
}


class SourceCitation(BaseModel):
    """A source URL and optional evidence snippet used in the final report."""

    url: str
    snippet: str | None = None
    title: str | None = None


class ReportedClaim(BaseModel):
    """A claim ready for final report rendering."""

    claim_text: str
    original_sentence: str
    original_index: int = 0
    verdict: ReportVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    sources: list[SourceCitation] = Field(default_factory=list)
    credibility_indicator: str = ""

    @model_validator(mode="after")
    def _set_default_indicator(self) -> ReportedClaim:
        if not self.credibility_indicator:
            self.credibility_indicator = VERDICT_LABELS[self.verdict]
        return self


class ReportStatistics(BaseModel):
    """Numerical summary of a completed fact-check run."""

    total_claims: int
    supported: int
    refuted: int
    insufficient_evidence: int
    conflicting_evidence: int
    credibility_score: float = 0.0
    credibility_label: str = "Unknown"

    @model_validator(mode="after")
    def _derive_credibility(self) -> ReportStatistics:
        if self.total_claims == 0:
            self.credibility_score = 0.0
            self.credibility_label = "No Claims Found"
            return self

        self.credibility_score = round((self.supported / self.total_claims) * 100, 1)
        uncertain = self.insufficient_evidence + self.conflicting_evidence

        if self.credibility_score >= 80:
            self.credibility_label = "High"
        elif self.credibility_score >= 50:
            self.credibility_label = "Medium"
        elif uncertain == self.total_claims:
            self.credibility_label = "Unverifiable"
        else:
            self.credibility_label = "Low"

        return self


class SummaryOutput(BaseModel):
    """Structured output expected from the summary LLM call."""

    summary: str


class FactCheckReport(BaseModel):
    """Internal report object rendered into the shared state's markdown string."""

    session_id: str
    original_text: str
    summary: str
    statistics: ReportStatistics
    claims: list[ReportedClaim]
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_used: str = ""
    generation_method: str = "template"
