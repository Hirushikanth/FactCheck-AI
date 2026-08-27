"""Ollama integration and health checks."""

from __future__ import annotations

import json
import time
from typing import Any

# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
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


async def probe_ollama_capabilities(
    settings: AppSettings,
    client: httpx.AsyncClient,
    *,
    show_raw_thinking: bool = False,
) -> dict[str, Any]:
    """Probe model availability and streamed thinking support without side effects.

    The caller owns ``client``. The probe only reads the model list and sends a
    small chat request; it never downloads or changes a model. Raw thinking is
    excluded unless explicitly requested for local diagnostics.
    """

    base_url = _base_url(settings)
    result: dict[str, Any] = {
        "status": "probe_failed",
        "reachable": False,
        "model_installed": False,
        "model": settings.ollama_model,
        "base_url": base_url,
        "thinking_chunks": 0,
        "content_chunks": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    try:
        tags_response = await client.get(f"{base_url}/api/tags")
        tags_response.raise_for_status()
        tags_payload = tags_response.json()
        model_names = {
            item.get("name")
            for item in tags_payload.get("models", [])
            if isinstance(item, dict)
        }
        result["reachable"] = True
        result["model_installed"] = settings.ollama_model in model_names
        if not result["model_installed"]:
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result

        thinking_parts: list[str] = []
        async with client.stream(
            "POST",
            f"{base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": "Reply with one word: ready."}],
                "stream": True,
                "think": True,
                "options": {"temperature": settings.ollama_temperature},
            },
        ) as chat_response:
            chat_response.raise_for_status()
            async for line in chat_response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                thinking = message.get("thinking")
                content = message.get("content")
                if isinstance(thinking, str) and thinking:
                    result["thinking_chunks"] += 1
                    thinking_parts.append(thinking)
                if isinstance(content, str) and content:
                    result["content_chunks"] += 1
                for metric in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                ):
                    if metric in chunk:
                        result[metric] = chunk[metric]

        result["status"] = (
            "supported" if result["thinking_chunks"] > 0 else "unsupported"
        )
        if show_raw_thinking:
            result["thinking"] = "".join(thinking_parts)
    except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
        # Keep diagnostics useful without exposing URLs, prompts, or model text.
        result["error"] = type(exc).__name__

    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


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
