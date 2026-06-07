# ADR-02: Mistral 7B via Ollama

## Status

Accepted for Phase 1.

## Decision

Use `mistral:7b` served by Ollama for local LLM inference.

## Rationale

The project proposal originally selected Qwen 2.5 3B, but development moved to Mistral 7B because Qwen was not reliable enough for structured verifier outputs. Mistral 7B remains practical for local or LAN-hosted Ollama inference while improving reasoning stability for the verifier.

## Consequences

The backend must never hardcode host details. `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are environment-driven.
