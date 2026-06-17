# ADR-02: Gemma via Ollama

## Status

Accepted for Phase 1. Current default: `gemma4` (v0.6.0).

## Decision

Use `gemma4` served by Ollama for local LLM inference.

## Rationale

The project needs a model that produces reliable structured outputs for the verifier while remaining practical for local or LAN-hosted Ollama inference on academic hardware.

## Model history

1. **Qwen 2.5 3B** — original proposal selection; dropped because structured verifier outputs were not reliable enough.
2. **Mistral 7B** — interim default during early development; improved reasoning stability for the verifier.
3. **`gemma4`** — current default; better structured-output behaviour for extractor and verifier workloads in this codebase.

## Consequences

The backend must never hardcode host details. `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are environment-driven.
