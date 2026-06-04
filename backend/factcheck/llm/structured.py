"""Structured LLM invocation helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from json import JSONDecodeError
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from factcheck.llm.concurrency import get_ollama_semaphore


logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
MessageLike = tuple[str, str]


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return str(content)


def _parse_json_object(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text)
    except JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _structured_llm(llm: BaseChatModel, output_class: type[M]):
    # Ollama JSON mode (format="json"); schema shape still comes from prompts/retries.
    return llm.with_structured_output(output_class, method="json_mode")


async def call_llm_with_structured_output(
    *,
    llm: BaseChatModel,
    output_class: type[M],
    messages: Sequence[MessageLike],
    context_desc: str = "",
) -> M | None:
    """Call a chat model and parse its response into a Pydantic model."""

    try:
        async with get_ollama_semaphore():
            response = await _structured_llm(llm, output_class).ainvoke(list(messages))
        if response is None:
            raise ValueError("structured output returned None")
        return response
    except Exception as exc:
        logger.warning("Structured LLM call failed for %s: %s", context_desc or "request", exc)

    schema_hint = json.dumps(output_class.model_json_schema(), indent=2)
    retry_messages = list(messages) + [
        (
            "human",
            "Retry the previous task and return output matching this JSON schema exactly:\n"
            f"{schema_hint}",
        )
    ]
    try:
        async with get_ollama_semaphore():
            response = await _structured_llm(llm, output_class).ainvoke(retry_messages)
        if response is None:
            raise ValueError("structured output retry returned None")
        return response
    except Exception as exc:
        logger.warning(
            "Structured LLM retry failed for %s: %s",
            context_desc or "request",
            exc,
        )

        compact_schema = json.dumps(output_class.model_json_schema(), separators=(",", ":"))
        fallback_messages = list(messages) + [
            (
                "human",
                "Return only one JSON object that matches this schema exactly. "
                "Do not include markdown or explanation.\n"
                f"Schema: {compact_schema}",
            )
        ]
        try:
            async with get_ollama_semaphore():
                diagnostic_response = await llm.ainvoke(fallback_messages)
            diagnostic_text = _message_text(diagnostic_response)
            diagnostic_json = _parse_json_object(diagnostic_text)
            if diagnostic_json is None:
                return None

            return output_class.model_validate(diagnostic_json)
        except (Exception, ValidationError) as diagnostic_exc:
            logger.warning(
                "Structured LLM plain JSON fallback failed for %s: %s",
                context_desc or "request",
                diagnostic_exc,
            )
        return None
