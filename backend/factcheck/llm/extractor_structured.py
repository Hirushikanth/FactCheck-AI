"""Structured LLM invocation tuned for extractor stages on Gemma 4."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from json import JSONDecodeError
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from factcheck.llm.concurrency import get_ollama_semaphore
from factcheck.llm.structured import MessageLike, _message_text, _parse_json_object


logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# String fields that may appear last in extractor JSON and contain unescaped quotes.
_TRAILING_STRING_FIELDS = (
    "reasoning",
    "processed_sentence",
    "disambiguated_sentence",
    "claim",
)


def _structured_llm(llm: BaseChatModel, output_class: type[M]):
    return llm.with_structured_output(output_class, method="json_mode")


def _repair_trailing_string_field(text: str, field_name: str) -> str:
    """Escape unescaped double quotes inside a trailing JSON string field."""

    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*"', re.DOTALL)
    match = pattern.search(text)
    if not match:
        return text

    value_start = match.end()
    end_marker = text.rfind('"}')
    if end_marker <= value_start:
        end_marker = text.rfind('"')
        if end_marker <= value_start:
            return text

    inner = text[value_start:end_marker]
    escaped = (
        inner.replace('\\"', "__ESCAPED_QUOTE__")
        .replace('"', '\\"')
        .replace("__ESCAPED_QUOTE__", '\\"')
    )
    return text[:value_start] + escaped + text[end_marker:]


def _repair_json_string_values(text: str) -> str:
    """Attempt to fix unescaped double quotes inside JSON string values."""

    repaired = text
    for field_name in _TRAILING_STRING_FIELDS:
        candidate = _repair_trailing_string_field(repaired, field_name)
        if candidate != repaired:
            try:
                json.loads(candidate)
                return candidate
            except JSONDecodeError:
                repaired = candidate
    return repaired


def _parse_with_repair(text: str) -> dict[str, object] | None:
    parsed = _parse_json_object(text)
    if parsed is not None:
        return parsed

    repaired = _repair_json_string_values(text)
    if repaired != text:
        return _parse_json_object(repaired)
    return None


def _validate_parsed(output_class: type[M], parsed: dict[str, object]) -> M:
    return output_class.model_validate(parsed)


async def _plain_json_invoke(
    llm: BaseChatModel,
    messages: Sequence[MessageLike],
    output_class: type[M],
) -> M | None:
    async with get_ollama_semaphore():
        response = await llm.ainvoke(list(messages))
    text = _message_text(response)
    parsed = _parse_with_repair(text)
    if parsed is None:
        return None
    return _validate_parsed(output_class, parsed)


async def _json_mode_invoke(
    llm: BaseChatModel,
    messages: Sequence[MessageLike],
    output_class: type[M],
) -> M | None:
    async with get_ollama_semaphore():
        response = await _structured_llm(llm, output_class).ainvoke(list(messages))
    if response is None:
        return None
    return response


async def call_extractor_structured_output(
    *,
    llm: BaseChatModel,
    output_class: type[M],
    messages: Sequence[MessageLike],
    context_desc: str = "",
) -> M | None:
    """Call an extractor LLM with plain JSON first, then schema retry, then json_mode."""

    label = context_desc or "request"

    try:
        result = await _plain_json_invoke(llm, messages, output_class)
        if result is not None:
            logger.debug("Extractor structured call succeeded (plain JSON) for %s", label)
            return result
    except (ValidationError, ValueError, TypeError) as exc:
        logger.debug("Extractor plain JSON parse failed for %s: %s", label, exc)

    compact_schema = json.dumps(output_class.model_json_schema(), separators=(",", ":"))
    retry_messages = list(messages) + [
        (
            "human",
            "Return only one JSON object matching this schema exactly. "
            "No markdown. No explanation.\n"
            f"Schema: {compact_schema}",
        )
    ]
    try:
        result = await _plain_json_invoke(llm, retry_messages, output_class)
        if result is not None:
            logger.debug("Extractor structured call succeeded (schema retry) for %s", label)
            return result
    except (ValidationError, ValueError, TypeError) as exc:
        logger.debug("Extractor schema retry parse failed for %s: %s", label, exc)

    try:
        result = await _json_mode_invoke(llm, retry_messages, output_class)
        if result is not None:
            logger.debug("Extractor structured call succeeded (json_mode fallback) for %s", label)
            return result
    except Exception as exc:
        logger.warning("Extractor json_mode fallback failed for %s: %s", label, exc)

    logger.warning("Extractor structured call failed for %s after all tiers", label)
    return None
