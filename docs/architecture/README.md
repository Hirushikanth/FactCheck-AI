# Architecture

This directory contains the architecture artifacts for FactCheck AI.

## Documents

- [`system-overview.md`](system-overview.md) — condensed system overview and component topology.
- [`state-schema.md`](state-schema.md) — shared state schema used by all agents.
- [`api-and-sse-contract.md`](api-and-sse-contract.md) — implemented REST and SSE contract.

## Implementation Status

| Component | Status |
|---|---|
| LangGraph pipeline (orchestrator → extractor → verifier → reporter) | Implemented |
| Dialogue agent (standalone graph, on-demand) | Implemented |
| SQLite session persistence | Implemented |
| REST API + SSE streaming | Implemented |
| React TypeScript frontend | Planned |

The backend API is at version `0.6.0`. Clients can create sessions, stream pipeline progress, retrieve results, and send follow-up dialogue messages. A production frontend is not yet in the repository.
