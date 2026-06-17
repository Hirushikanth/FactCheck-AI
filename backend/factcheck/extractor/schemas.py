"""Internal data models for the claim extractor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ContextualSentence(BaseModel):
    """A sentence paired with its local source context."""

    original_sentence: str
    context_for_llm: str
    metadata: str | None = None
    original_index: int


class SelectedContent(BaseModel):
    """Sentence content selected as potentially verifiable."""

    processed_sentence: str
    original_context_item: ContextualSentence
    preceding_context_item: ContextualSentence


class DisambiguatedContent(BaseModel):
    """Selected content with resolvable ambiguity removed."""

    disambiguated_sentence: str
    original_selected_item: SelectedContent


class PotentialClaim(BaseModel):
    """A decomposed factual claim before final validation."""

    claim_text: str
    disambiguated_sentence: str
    original_sentence: str
    original_index: int
    fidelity_status: Literal["faithful", "fallback"] = "faithful"


class ValidatedClaim(BaseModel):
    """A claim that is complete enough to pass to the verifier."""

    claim_text: str
    is_complete_declarative: bool
    disambiguated_sentence: str
    original_sentence: str
    original_index: int
    fidelity_status: Literal["faithful", "fallback"] = "faithful"


class ExtractorStageFailure(BaseModel):
    """Records a sentence dropped during an extractor LLM stage."""

    stage: Literal["selection", "disambiguation", "decomposition"]
    sentence: str
    reason: Literal["voting_failed", "parse_failed", "no_output"]
    successes: int
    attempts: int


class ExtractorState(BaseModel):
    """State object used inside the extractor subgraph."""

    raw_input: str
    contextual_sentences: list[ContextualSentence] = Field(default_factory=list)
    preceding_context_sentences: list[ContextualSentence] = Field(default_factory=list)
    selected_contents: list[SelectedContent] = Field(default_factory=list)
    disambiguated_contents: list[DisambiguatedContent] = Field(default_factory=list)
    potential_claims: list[PotentialClaim] = Field(default_factory=list)
    validated_claims: list[ValidatedClaim] = Field(default_factory=list)
    stage_failures: list[ExtractorStageFailure] = Field(default_factory=list)
    metadata: str | None = None
    extraction_mode: Literal["auto", "claim", "document"] = "auto"
    resolved_extraction_mode: Literal["direct_claim", "document"] | None = None
