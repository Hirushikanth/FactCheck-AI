"""Query generator node for verifier search."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from factcheck.llm.factory import get_verifier_llm
from factcheck.llm.structured import call_llm_with_structured_output
from factcheck.verifier.config import MAX_SEARCH_QUERIES, QUERY_GEN_TEMPERATURE
from factcheck.verifier.prompts import (
    QUERY_GENERATOR_HUMAN_PROMPT,
    QUERY_GENERATOR_SYSTEM_PROMPT,
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


def _clean_queries(queries: list[str], claim: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    normalized_claim = " ".join(claim.strip().rstrip(".?!").split())
    for query in queries:
        normalized = " ".join(query.strip().rstrip(".?!").split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= MAX_SEARCH_QUERIES:
            break

    if not cleaned:
        cleaned.append(normalized_claim or claim)

    if len(cleaned) == 1 and cleaned[0].casefold() == normalized_claim.casefold():
        for expanded in (
            _keyword_query(normalized_claim),
            f"{normalized_claim} fact check".strip(),
        ):
            key = expanded.casefold()
            if expanded and key not in seen:
                seen.add(key)
                cleaned.append(expanded)
            if len(cleaned) >= MAX_SEARCH_QUERIES:
                break

    return cleaned


async def query_generator_node(state: VerifierState) -> dict[str, list[str]]:
    """Generate targeted search queries for one claim."""

    if state.claim_result is not None:
        return {"search_queries": state.search_queries}

    llm = get_verifier_llm(temperature=QUERY_GEN_TEMPERATURE)
    response = await call_llm_with_structured_output(
        llm=llm,
        output_class=QueryGeneratorOutput,
        messages=[
            ("system", QUERY_GENERATOR_SYSTEM_PROMPT),
            (
                "human",
                QUERY_GENERATOR_HUMAN_PROMPT.format(
                    claim=state.claim,
                    max_queries=MAX_SEARCH_QUERIES,
                ),
            ),
        ],
        context_desc=f"query generation for '{state.claim}'",
    )

    return {"search_queries": _clean_queries(response.queries if response else [], state.claim)}
