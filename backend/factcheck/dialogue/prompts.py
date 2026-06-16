"""All prompt templates and helper functions for the Dialogue Agent.

Designed for Mistral 7B Instruct via Ollama.  Every prompt is
token-budget conscious and written to minimise hallucination from
model training data.

Key design constraints:
- Intent classification uses single-word output (no JSON) — more reliable
  on 7B models and uses 5–10× fewer output tokens.
- Grounding instruction is duplicated at the end of the user message
  (recency bias mitigation: small models attend more strongly to the
  last ~200 tokens of context).
- Query rewriting has a regex bypass heuristic — if no coreference
  patterns are detected, the call is skipped to save ~580 tokens.
"""

from __future__ import annotations

import re
from typing import Optional

from factcheck.dialogue.config import MAX_FC_CONTEXT_TOKENS, SYSTEM_PROMPT_TOKENS
from factcheck.dialogue.schemas import DialogueTurn


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (~380 tokens — see SYSTEM_PROMPT_TOKENS in config.py)
# ─────────────────────────────────────────────────────────────────────────────

DIALOGUE_SYSTEM_PROMPT = """You are a conversational fact-checking assistant.
Your role is to help users understand the results of a completed fact-check session.

RULES — follow these without exception:
1. GROUNDING: Answer ONLY from the [FACT-CHECK CONTEXT] block provided.
   Never use your own knowledge or training data to answer factual questions.
2. HONESTY: If information is not in the context, say "I don't have that
   information in this session's results" and suggest submitting a new claim.
3. CLARITY: Explain verdicts (SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE /
   CONFLICTING_EVIDENCE), confidence scores, and evidence in plain language.
4. CONCISENESS: Keep responses under 250 words unless the user asks for detail.
5. NO SPECULATION: Do not speculate about why a claim might be true or false
   beyond what the evidence in the context says.
6. SCOPE: You handle questions about fact-check results only. Politely decline
   unrelated requests and explain what you can help with.

When you see [FACT-CHECK CONTEXT], treat it as the authoritative source of
truth for this conversation."""


# ─────────────────────────────────────────────────────────────────────────────
# FACT-CHECK CONTEXT COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def compress_factcheck_context(
    claim_results: list[dict],
    *,
    max_claim_chars: int = 120,
    max_evidence_chars: int = 200,
    max_sources: int = 3,
) -> str:
    """Build a compact, structured reference block from ClaimResult dicts.

    Target: ≤ 800 tokens for up to 10 claims.

    Each entry uses the project's ClaimResult field names:
      - ``claim``      — the claim text
      - ``verdict``    — SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE / CONFLICTING_EVIDENCE
      - ``confidence`` — 0.0–1.0 float
      - ``evidence``   — list[str]; first 1-2 items joined for the summary
      - ``sources``    — list[str]
    """
    if not claim_results:
        return "=== FACT-CHECK RESULTS (session context) ===\nNo claims were checked in this session.\n=== END OF FACT-CHECK CONTEXT ==="

    lines = ["=== FACT-CHECK RESULTS (session context) ==="]

    for i, cr in enumerate(claim_results, 1):
        # Claim text
        claim_text = cr.get("claim", "Unknown claim")
        claim_short = claim_text[:max_claim_chars]
        if len(claim_text) > max_claim_chars:
            claim_short += "..."

        # Verdict
        verdict = cr.get("verdict", "UNKNOWN")

        # Confidence: stored as 0.0–1.0 float in ClaimResult
        confidence_raw = cr.get("confidence", None)
        if isinstance(confidence_raw, float):
            confidence_str = f"{confidence_raw:.0%}"
        elif confidence_raw is not None:
            confidence_str = str(confidence_raw)
        else:
            confidence_str = "N/A"

        # Evidence summary: join first 2 evidence snippets + reasoning excerpt
        evidence_items: list[str] = cr.get("evidence", [])
        reasoning: str = cr.get("reasoning", "")
        if evidence_items:
            evidence_raw = "; ".join(evidence_items[:2])
        elif reasoning:
            evidence_raw = reasoning
        else:
            evidence_raw = "No evidence available."

        evidence_short = evidence_raw[:max_evidence_chars]
        if len(evidence_raw) > max_evidence_chars:
            evidence_short += "..."

        # Sources
        sources_list: list[str] = cr.get("sources", [])[:max_sources]
        sources_str = ", ".join(sources_list) if sources_list else "No sources listed"

        lines.append(
            f"\n[Claim {i}] {claim_short}\n"
            f"  Verdict: {verdict} (confidence: {confidence_str})\n"
            f"  Evidence: {evidence_short}\n"
            f"  Sources: {sources_str}"
        )

    lines.append("\n=== END OF FACT-CHECK CONTEXT ===")
    return "\n".join(lines)


def compress_factcheck_runs(
    runs: list[dict],
    *,
    max_input_chars: int = 120,
) -> str:
    """Build cumulative context from multiple completed fact-check runs."""
    if not runs:
        return compress_factcheck_context([])

    sections: list[str] = []
    for run in runs:
        sequence = run.get("sequence", 0)
        raw_input = run.get("raw_input", "")
        input_short = raw_input[:max_input_chars]
        if len(raw_input) > max_input_chars:
            input_short += "..."

        claim_block = compress_factcheck_context(run.get("claim_results", []))
        claim_block = claim_block.replace(
            "=== FACT-CHECK RESULTS (session context) ===",
            "",
        ).replace("=== END OF FACT-CHECK CONTEXT ===", "").strip()

        sections.append(
            f"=== FACT-CHECK RUN {sequence} ===\n"
            f"Input: {input_short}\n\n"
            f"{claim_block}"
        )

    return (
        "=== FACT-CHECK RESULTS (session context) ===\n"
        + "\n\n".join(sections)
        + "\n=== END OF FACT-CHECK CONTEXT ==="
    )


def build_session_context_extras(
    original_text: str | None,
    final_report: str | None,
    *,
    max_original_tokens: int = 80,
    max_report_tokens: int = 200,
) -> str:
    """Build optional session metadata appended to the fact-check context block."""
    from factcheck.dialogue.utils.tokens import estimate_tokens, truncate_to_tokens

    sections: list[str] = []

    if original_text and original_text.strip():
        snippet = original_text.strip()
        if estimate_tokens(snippet) > max_original_tokens:
            snippet = truncate_to_tokens(snippet, max_original_tokens)
        sections.append(f"[ORIGINAL TEXT SUBMITTED]\n{snippet}")

    if final_report and final_report.strip():
        snippet = final_report.strip()
        if estimate_tokens(snippet) > max_report_tokens:
            snippet = truncate_to_tokens(snippet, max_report_tokens)
        sections.append(f"[FACT-CHECK REPORT EXCERPT]\n{snippet}")

    if not sections:
        return ""

    combined = "\n\n".join(sections)
    if estimate_tokens(combined) > MAX_FC_CONTEXT_TOKENS:
        combined = truncate_to_tokens(combined, MAX_FC_CONTEXT_TOKENS)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTENTS = frozenset({
    "clarification",
    "general_question",
    "new_claim",
    "out_of_scope",
    "ask_clarification",
})

FALLBACK_INTENT = "clarification"


def build_intent_prompt(
    user_message: str,
    num_claims: int,
    topic_keywords: str,
) -> str:
    """Build the single-word intent classification prompt.

    The ``LABEL:`` suffix at the very end steers Mistral 7B to continue
    with the label word and stop — avoiding preamble sentences.
    """
    return (
        "Classify the message below into exactly one category. Reply with ONE WORD only.\n\n"
        "Categories:\n"
        "- clarification    (asking about fact-check results, verdicts, evidence, sources)\n"
        "- general_question (factual question about the topic, not directly about check results)\n"
        "- new_claim        (submitting new text or a statement to be fact-checked)\n"
        "- out_of_scope     (completely unrelated to fact-checking or this session)\n"
        "- ask_clarification (too vague to classify — need more context from the user)\n\n"
        f"Session: {num_claims} claim(s) checked. Topic keywords: {topic_keywords}\n\n"
        f"Message: {user_message[:300]}\n\n"
        "LABEL:"
    )


def parse_intent(raw_output: str) -> str:
    """Parse and validate intent from raw LLM output.

    Checks for known intent labels anywhere in the output (handles preamble),
    then falls back to the first token. Unknown outputs use FALLBACK_INTENT.
    """
    if not raw_output or not raw_output.strip():
        return FALLBACK_INTENT

    cleaned = raw_output.strip().lower()

    for intent in sorted(VALID_INTENTS, key=len, reverse=True):
        if intent in cleaned:
            return intent

    first_word = cleaned.split()[0].rstrip(".,;:!?")
    return first_word if first_word in VALID_INTENTS else FALLBACK_INTENT


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER
# ─────────────────────────────────────────────────────────────────────────────

# Coreference patterns that indicate the query cannot stand alone.
# If none match, the rewrite LLM call is skipped (~580 tokens saved per turn).
_COREFERENCE_PATTERNS = [
    re.compile(r"\b(it|this|that|they|these|those)\b", re.IGNORECASE),
    re.compile(r"\b(the one|the claim|the result|the verdict|the source|the evidence)\b", re.IGNORECASE),
    re.compile(r"\b(there|then|same|another one|the other)\b", re.IGNORECASE),
]


def needs_rewriting(message: str) -> bool:
    """Heuristic: does *message* contain references that need resolution?

    Returns True if any coreference pattern matches, indicating the query
    depends on conversation history to be understood.
    """
    for pattern in _COREFERENCE_PATTERNS:
        if pattern.search(message):
            return True
    return False


def build_rewriter_prompt(
    current_message: str,
    recent_history: list[DialogueTurn],
    summary: Optional[str],
) -> str:
    """Build the query rewriting prompt.

    Uses at most the last 4 turns (to keep rewriter context tight; ~300 tokens).
    The model is instructed to return ONLY the rewritten message.
    """
    history_text = ""
    for turn in recent_history[-4:]:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        snippet = turn["content"][:250]
        history_text += f"{prefix}: {snippet}\n"

    summary_line = f"Earlier summary: {summary}\n\n" if summary else ""

    return (
        "Rewrite the CURRENT MESSAGE to be self-contained and clear.\n"
        "Resolve any pronouns or vague references using the conversation context.\n"
        "Rules:\n"
        "- Do NOT answer the question.\n"
        "- Do NOT add information that is not in the conversation.\n"
        "- Return ONLY the rewritten message. Max 2 sentences.\n\n"
        f"{summary_line}"
        f"Recent conversation:\n{history_text}\n"
        f"CURRENT MESSAGE: {current_message}\n\n"
        "REWRITTEN:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY COMPRESSOR
# ─────────────────────────────────────────────────────────────────────────────

def build_compressor_prompt(
    turns_to_compress: list[DialogueTurn],
    existing_summary: Optional[str],
) -> str:
    """Build the rolling history compression prompt.

    Produces a 2-sentence summary (max 80 words) covering what was asked
    and what key answers were given.
    """
    turns_text = ""
    for turn in turns_to_compress:
        prefix = "User" if turn["role"] == "user" else "Bot"
        turns_text += f"{prefix}: {turn['content'][:300]}\n"

    existing_section = (
        f"Existing summary (incorporate into your new summary):\n{existing_summary}\n\n"
        if existing_summary
        else ""
    )

    return (
        "Write a 2-sentence factual summary (max 80 words) of the conversation below.\n"
        "Include: what questions were asked and what key answers were given.\n"
        "Do NOT speculate or add interpretation. Return only the summary text.\n\n"
        f"{existing_section}"
        f"Conversation:\n{turns_text}\n\n"
        "SUMMARY:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# Intent-specific instruction appended to the user message.
# Placed at the end of the prompt for recency-bias effect in small models.
INTENT_HINTS: dict[str, str] = {
    "clarification": "",
    "general_question": (
        "\n[INSTRUCTION: Answer only if the information is available in the "
        "[FACT-CHECK CONTEXT]. If not, say you don't have it and suggest "
        "submitting a new claim for fact-checking.]"
    ),
    "out_of_scope": (
        "\n[INSTRUCTION: Politely decline this request. Explain that you can "
        "only assist with questions about the current fact-check session results.]"
    ),
    "ask_clarification": (
        "\n[INSTRUCTION: Ask ONE specific clarifying question to understand "
        "what the user wants to know. Do not attempt to answer.]"
    ),
}

GROUNDING_REMINDER = "\n\n[REMINDER: Use ONLY the information from [FACT-CHECK CONTEXT] above.]"


def build_generator_messages(
    fc_context_block: str,
    summary: Optional[str],
    windowed_history: list[DialogueTurn],
    rewritten_query: str,
    intent: str,
) -> list[dict]:
    """Assemble the full messages list for the ChatOllama call.

    The system message embeds the fact-check context block so it is always
    present regardless of how the history window is trimmed.

    The grounding reminder is appended to the final user message to exploit
    the recency bias of small transformer models.
    """
    # System message with embedded context
    system_content = DIALOGUE_SYSTEM_PROMPT
    system_content += f"\n\n[FACT-CHECK CONTEXT]\n{fc_context_block}"
    if summary:
        system_content += f"\n\n[CONVERSATION SUMMARY]\n{summary}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject windowed history as chat turns
    for turn in windowed_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Build the final user message with intent hint + grounding reminder
    intent_hint = INTENT_HINTS.get(intent, "")
    final_user_content = rewritten_query + intent_hint + GROUNDING_REMINDER

    messages.append({"role": "user", "content": final_user_content})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# NEW CLAIM ACKNOWLEDGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def new_claim_acknowledgement_prompt(claim_text: str) -> str:
    """Prompt for a friendly new-claim-queued acknowledgement message."""
    return (
        f"The user has submitted a new claim for fact-checking: '{claim_text[:200]}'\n"
        "Write a brief, friendly acknowledgement (2-3 sentences) telling the user:\n"
        "1. Their new claim has been queued for fact-checking.\n"
        "2. Results will appear shortly.\n"
        "3. They can continue asking about the previous results in the meantime.\n\n"
        "ACKNOWLEDGEMENT:"
    )
