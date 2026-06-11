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


def get_verifier_llm(
    *,
    temperature: float,
    num_ctx: int | None = None,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create an Ollama chat model for verifier stages."""

    resolved_settings = settings or get_settings()
    kwargs = {
        "base_url": _base_url(resolved_settings),
        "model": resolved_settings.ollama_model,
        "temperature": temperature,
    }
    if num_ctx is not None and resolved_settings.ollama_num_ctx is not None:
        resolved_num_ctx = min(num_ctx, resolved_settings.ollama_num_ctx)
    else:
        resolved_num_ctx = num_ctx or resolved_settings.ollama_num_ctx
    if resolved_num_ctx is not None:
        kwargs["num_ctx"] = resolved_num_ctx
    return ChatOllama(**kwargs)


def get_reporter_llm(
    *,
    temperature: float,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create an Ollama chat model for reporter summary generation."""

    resolved_settings = settings or get_settings()
    kwargs = {
        "base_url": _base_url(resolved_settings),
        "model": resolved_settings.ollama_model,
        "temperature": temperature,
    }
    if num_ctx is not None and resolved_settings.ollama_num_ctx is not None:
        resolved_num_ctx = min(num_ctx, resolved_settings.ollama_num_ctx)
    else:
        resolved_num_ctx = num_ctx or resolved_settings.ollama_num_ctx
    if resolved_num_ctx is not None:
        kwargs["num_ctx"] = resolved_num_ctx
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    return ChatOllama(**kwargs)
