"""LLM factories for agent-specific runtime settings."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from factcheck.config import AppSettings, get_settings
from factcheck.dialogue.config import (
    ACKNOWLEDGE_LLM_PARAMS,
    CLASSIFIER_LLM_PARAMS,
    COMPRESSOR_LLM_PARAMS,
    DIALOGUE_LLM_PARAMS,
    REWRITER_LLM_PARAMS,
)
from factcheck.llm.ollama import _base_url


def get_extractor_llm(
    *,
    temperature: float,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create an Ollama chat model for extractor stages."""

    resolved_settings = settings or get_settings()
    kwargs = {
        "base_url": _base_url(resolved_settings),
        "model": resolved_settings.ollama_model,
        "temperature": temperature,
        "client_kwargs": {"timeout": resolved_settings.ollama_timeout},
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
        "client_kwargs": {"timeout": resolved_settings.ollama_timeout},
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
        "client_kwargs": {"timeout": resolved_settings.ollama_timeout},
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


# ─────────────────────────────────────────────────────────────────────────────
# Dialogue Agent LLM factories
# Each function creates a fresh ChatOllama instance with parameters from
# factcheck.dialogue.config.  All use num_ctx=8192 explicitly to override
# Ollama's default 4096 ceiling.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_dialogue_num_ctx(settings: AppSettings, requested: int) -> int:
    """Return the effective num_ctx for dialogue calls (cap at env setting)."""
    if settings.ollama_num_ctx is not None:
        return min(requested, settings.ollama_num_ctx)
    return requested


def _dialogue_chat_ollama(
    params: dict,
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Build a ChatOllama instance for dialogue nodes from config params."""
    resolved_settings = settings or get_settings()
    num_ctx = _resolve_dialogue_num_ctx(resolved_settings, params["num_ctx"])
    return ChatOllama(
        base_url=_base_url(resolved_settings),
        model=resolved_settings.ollama_model,
        num_ctx=num_ctx,
        num_predict=params.get("num_predict"),
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        repeat_penalty=params.get("repeat_penalty"),
        stop=params.get("stop"),
        client_kwargs={"timeout": resolved_settings.ollama_timeout},
    )


def get_dialogue_llm(
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create the main dialogue response generation LLM."""
    return _dialogue_chat_ollama(DIALOGUE_LLM_PARAMS, settings=settings)


def get_dialogue_classifier_llm(
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create the intent classification LLM."""
    return _dialogue_chat_ollama(CLASSIFIER_LLM_PARAMS, settings=settings)


def get_dialogue_rewriter_llm(
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create the query rewriting LLM."""
    return _dialogue_chat_ollama(REWRITER_LLM_PARAMS, settings=settings)


def get_dialogue_compressor_llm(
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create the history compression LLM."""
    return _dialogue_chat_ollama(COMPRESSOR_LLM_PARAMS, settings=settings)


def get_dialogue_acknowledge_llm(
    *,
    settings: AppSettings | None = None,
) -> ChatOllama:
    """Create the new-claim acknowledgement LLM."""
    return _dialogue_chat_ollama(ACKNOWLEDGE_LLM_PARAMS, settings=settings)
