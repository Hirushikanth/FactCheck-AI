"""Server-Sent Events formatting utilities."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Recursively convert Pydantic models into JSON-serializable values."""

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame with compact, stable JSON."""

    payload = json.dumps(to_jsonable(data), separators=(",", ":"), sort_keys=True)
    return f"event: {event}\ndata: {payload}\n\n"
