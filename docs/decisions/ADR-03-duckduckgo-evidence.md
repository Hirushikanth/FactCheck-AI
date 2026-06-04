# ADR-03: DuckDuckGo as Evidence Retrieval Backend

## Status

Accepted for later implementation. Amended by ADR-05 to keep DuckDuckGo as the primary provider while adding optional Tavily and Serper fallbacks.

## Decision

Use DuckDuckGo search for live web evidence retrieval in the Verifier Agent.

## Rationale

DuckDuckGo search is free, requires no account, and avoids paid cloud API dependencies. This supports the project goal of a locally runnable academic prototype with no ongoing cost.

## Phase 1 Boundary

Phase 1 documents this decision only. Search integration begins in Phase 3.
