"""LLM factories for agent-specific runtime settings."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from factcheck.config import AppSettings, get_settings
from factcheck.llm.ollama import _base_url


def get_extractor_llm(
    *,
    temperature: float,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create an Ollama chat model for extractor stages."""

    resolved_settings = settings or get_settings()
    return ChatOllama(
        base_url=_base_url(resolved_settings),
        model=resolved_settings.ollama_model,
        temperature=temperature,
    )
