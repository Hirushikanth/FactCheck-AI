"""Query generator node for verifier search."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.verifier.config import QUERY_GEN_NUM_CTX, QUERY_GEN_TEMPERATURE, QUERIES_PER_ITERATION
from factcheck.verifier.prompts import (
    QUERY_GENERATOR_INITIAL_HUMAN_PROMPT,
    QUERY_GENERATOR_INITIAL_SYSTEM_PROMPT,
    QUERY_GENERATOR_ITERATIVE_HUMAN_PROMPT,
    QUERY_GENERATOR_ITERATIVE_SYSTEM_PROMPT,
)
from factcheck.verifier.schemas import VerifierState


class QueryGeneratorOutput(BaseModel):
    """Structured output for query generation."""

    queries: list[str] = Field(default_factory=list)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _keyword_query(text: str) -> str:
    tokens = [
        token
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _QUERY_STOPWORDS
    ]
    return " ".join(tokens)


def _clean_query(queries: list[str], claim: str, previous_queries: list[str]) -> str | None:
    cleaned: list[str] = []
    seen: set[str] = {query.casefold() for query in previous_queries}
    normalized_claim = " ".join(claim.strip().rstrip(".?!").split())
    for query in queries:
        normalized = " ".join(query.strip().rstrip(".?!").split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= QUERIES_PER_ITERATION:
            break

    for expanded in (
        _keyword_query(normalized_claim),
        f"{normalized_claim} fact check".strip(),
        normalized_claim or claim,
    ):
        if not cleaned:
            key = expanded.casefold()
            if expanded and key not in seen:
                seen.add(key)
                cleaned.append(expanded)
            if len(cleaned) >= QUERIES_PER_ITERATION:
                break

    return cleaned[0] if cleaned else None


def _iterative_missing_aspects(state: VerifierState) -> list[str]:
    if state.intermediate_assessment and state.intermediate_assessment.missing_aspects:
        return state.intermediate_assessment.missing_aspects
    return ["independent evidence that directly addresses the claim"]


def _query_messages(state: VerifierState) -> list[tuple[str, str]]:
    source_sentence = state.source_sentence or state.claim_text
    if state.iteration_count <= 0:
        return [
            ("system", QUERY_GENERATOR_INITIAL_SYSTEM_PROMPT),
            (
                "human",
                QUERY_GENERATOR_INITIAL_HUMAN_PROMPT.format(
                    source_sentence=source_sentence,
                    claim=state.claim_text,
                ),
            ),
        ]

    previous_queries = "\n".join(f"- {query}" for query in state.all_queries) or "- None"
    missing_aspects = "\n".join(f"- {aspect}" for aspect in _iterative_missing_aspects(state))
    return [
        ("system", QUERY_GENERATOR_ITERATIVE_SYSTEM_PROMPT),
        (
            "human",
            QUERY_GENERATOR_ITERATIVE_HUMAN_PROMPT.format(
                source_sentence=source_sentence,
                claim=state.claim_text,
                previous_queries=previous_queries,
                missing_aspects=missing_aspects,
            ),
        ),
    ]


async def query_generator_node(state: VerifierState) -> dict[str, str | list[str] | None]:
    """Generate targeted search queries for one claim."""

    if state.claim_result is not None:
        return {"current_query": state.current_query, "all_queries": state.all_queries}

    llm = get_verifier_llm(temperature=QUERY_GEN_TEMPERATURE, num_ctx=QUERY_GEN_NUM_CTX)
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=QueryGeneratorOutput,
        messages=_query_messages(state),
        context_desc=f"query generation for '{state.claim_text}'",
    )

    query = _clean_query(response.queries if response else [], state.claim_text, state.all_queries)
    if query is None:
        return {"current_query": None}

    return {
        "current_query": query,
        "all_queries": state.all_queries + [query],
    }
