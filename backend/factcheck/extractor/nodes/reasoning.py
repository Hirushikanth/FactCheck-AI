"""Shared schema helpers for structured extractor reasoning."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator


def _normalize_reasoning(value: object) -> object:
    if isinstance(value, list):
        return "\n".join(str(step) for step in value)
    return value


ReasoningText = Annotated[str, BeforeValidator(_normalize_reasoning)]
