"""Ollama integration and health checks."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_ollama import ChatOllama

from factcheck.config import AppSettings, get_settings


def _base_url(settings: AppSettings) -> str:
    return str(settings.ollama_base_url).rstrip("/")


def get_chat_model(settings: AppSettings | None = None) -> ChatOllama:
    """Create a configured ChatOllama instance for future agents."""

    resolved_settings = settings or get_settings()
    return ChatOllama(
        base_url=_base_url(resolved_settings),
        model=resolved_settings.ollama_model,
        temperature=resolved_settings.ollama_temperature,
    )


async def check_ollama_health(
    settings: AppSettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Check whether Ollama is reachable and the configured model is present."""

    resolved_settings = settings or get_settings()
    base_url = _base_url(resolved_settings)
    result: dict[str, Any] = {
        "reachable": False,
        "model_loaded": False,
        "base_url": base_url,
        "model": resolved_settings.ollama_model,
    }

    try:
        async with httpx.AsyncClient(
            timeout=resolved_settings.ollama_timeout,
            transport=transport,
        ) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        result["error"] = str(exc)
        return result

    model_names = {item.get("name") for item in payload.get("models", [])}
    result["reachable"] = True
    result["model_loaded"] = resolved_settings.ollama_model in model_names
    return result
