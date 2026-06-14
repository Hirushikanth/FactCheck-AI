"""TypedDict schemas for the Dialogue Agent.

These types are intentionally separate from FactCheckState — the dialogue
agent takes a snapshot of completed fact-check results at conversation start
and works from that stable snapshot throughout the session.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class DialogueTurn(TypedDict):
    """A single turn in the conversation history."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: float
    intent: Optional[str]  # classified intent — populated on user turns only
    token_estimate: int  # estimated tokens for budget tracking


class ConversationSummary(TypedDict):
    """Rolling compressed summary of older conversation turns."""

    text: str  # summary text, capped at MAX_SUMMARY_TOKENS
    turns_compressed: int  # how many turns are covered by this summary
    last_updated: float  # unix timestamp of the last compression run


class DialogueState(TypedDict):
    """Full state for the Dialogue Agent StateGraph.

    Fields are mapped from the project's FactCheckState / ClaimResult:
      claim_results[].claim         → verifications context
      claim_results[].verdict       → SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE / CONFLICTING_EVIDENCE
      claim_results[].confidence    → 0.0–1.0 float
      claim_results[].evidence[]    → joined for evidence_summary
      claim_results[].sources       → list[str]
      raw_input                     → original_text equivalent
      final_report                  → report_markdown equivalent
    """

    # ── Session ───────────────────────────────────────────────────────────────
    session_id: str

    # ── Fact-check context snapshot (read-only for dialogue) ─────────────────
    original_text: Optional[str]
    claim_results: list[dict]  # snapshot of ClaimResult dicts from FactCheckState
    final_report: Optional[str]

    # ── Compiled context (cached after init_context, reused each turn) ────────
    _compressed_fc_context: Optional[str]

    # ── Conversation history ──────────────────────────────────────────────────
    dialogue_history: list[DialogueTurn]
    conversation_summary: Optional[ConversationSummary]

    # ── Current turn ──────────────────────────────────────────────────────────
    current_user_message: str
    classified_intent: Optional[str]  # clarification|general_question|new_claim|out_of_scope|ask_clarification
    rewritten_query: Optional[str]
    dialogue_response: Optional[str]

    # ── Internal assembled prompt (not persisted) ─────────────────────────────
    _assembled_messages: Optional[list[dict]]

    # ── Token budget tracking ─────────────────────────────────────────────────
    estimated_context_tokens: int
    needs_compression: bool

    # ── Pipeline handoff ──────────────────────────────────────────────────────
    needs_new_factcheck: bool
    new_claim_text: Optional[str]

    # ── Error passthrough ─────────────────────────────────────────────────────
    error_message: Optional[str]


class DialogueOutput(TypedDict):
    """Return value of run_dialogue(). Caller owns persistence."""

    response: str
    intent: str
    dialogue_history: list[DialogueTurn]
    conversation_summary: Optional[ConversationSummary]
    compressed_fc_context: Optional[str]
    needs_new_factcheck: bool
    new_claim_text: Optional[str]
    error: Optional[str]
