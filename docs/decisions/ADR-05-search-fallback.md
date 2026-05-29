# ADR-05: Search Provider Fallback Chain

## Status

Accepted for Phase 2.

## Decision

Use DuckDuckGo as the primary evidence search provider and configure Tavily and Serper as optional fallbacks. The default provider order is `duckduckgo,tavily,serper`.

## Rationale

DuckDuckGo remains the no-cost default and supports the local-first academic prototype goal recorded in ADR-03. However, live web search can fail, throttle, or return empty results. Tavily and Serper provide API-backed fallbacks when the corresponding API keys are configured.

## Consequences

The Verifier Agent should call the shared search fallback layer in Phase 3 instead of binding directly to a single provider. Paid providers are skipped unless `TAVILY_API_KEY` or `SERPER_API_KEY` is present, so local development still works without paid API credentials.
