# Architecture

This directory contains the architecture artifacts for FactCheck AI.

## Documents

- `[system-overview.md](system-overview.md)` — condensed system overview and component topology.
- `[state-schema.md](state-schema.md)` — frozen shared state schema for all later agents.
- `[api-and-sse-contract.md](api-and-sse-contract.md)` — REST and SSE contract planned for Phase 6.

## Current Boundary

Phase 2 implements claim extraction and prepares the reusable search fallback layer. Evidence retrieval in the Verifier Agent, verdict generation, dialogue, persistence, SSE streaming, and the final frontend remain later phases.