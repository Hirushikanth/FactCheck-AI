# ADR-02: Qwen 2.5 3B via Ollama

## Status

Accepted for Phase 1.

## Decision

Use `qwen2.5:3b` served by Ollama for local LLM inference.

## Rationale

The model is small enough for consumer hardware while still capable of structured reasoning tasks. Ollama provides a simple local HTTP interface and supports both MacBook-local and LAN-hosted deployment.

## Consequences

The backend must never hardcode host details. `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are environment-driven.
