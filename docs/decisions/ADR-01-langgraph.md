# ADR-01: LangGraph as Agent Orchestration Framework

## Status

Accepted for Phase 1.

## Decision

Use LangGraph for the multi-agent state graph and supervisor-style orchestration.

## Rationale

LangGraph provides deterministic graph execution, explicit edges, conditional routing, and a shared state object. These properties fit an evidence-grounded fact-checking system where the processing path must be auditable.

## Alternatives Considered

- CrewAI
- AutoGen
- Custom Python function pipeline

## Consequences

All agents must read from and write to `FactCheckState`. Agent-to-agent calls should not bypass the graph.
