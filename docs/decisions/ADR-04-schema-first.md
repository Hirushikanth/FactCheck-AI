# ADR-04: Schema-First Development

## Status

Accepted for Phase 1.

## Decision

Define and freeze the shared `FactCheckState`, `ClaimResult`, and `PipelineStatus` contracts before implementing agent behavior.

## Rationale

Every agent depends on the same state object. Freezing the schema early reduces integration risk and lets later phases implement agents independently without changing shared contracts.

## Consequences

After Phase 1, any schema change requires a documented version update and review of all affected agents, API models, and frontend types.
