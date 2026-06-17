# ADR-03: DuckDuckGo as Evidence Retrieval Backend

## Status

Implemented (v0.6.0). Amended by ADR-05 to keep DuckDuckGo as the primary provider while adding optional Tavily and Serper fallbacks.

## Decision

Use DuckDuckGo search for live web evidence retrieval in the Verifier Agent.

## Rationale

DuckDuckGo search is free, requires no account, and avoids paid cloud API dependencies. This supports the project goal of a locally runnable academic prototype with no ongoing cost.

## Implementation

Search integration lives in `backend/factcheck/search/`. The verifier calls the shared fallback layer (`backend/factcheck/search/fallback.py`) with DuckDuckGo as the default provider.
