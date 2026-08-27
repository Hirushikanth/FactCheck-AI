# Resilient Ollama and SSE Observability Design

## Goal

Make fact-check runs resilient to transient local-Ollama failures and visibly
progressive in the frontend. When the configured model exposes streamed
thinking, users may opt in to see that exact model-emitted text. When it does
not, users see safe, evidence-grounded operational progress and summaries.

## Scope

This change retains the existing FastAPI, LangGraph, SQLite, and React/Vite
architecture. It does not add a cloud LLM, download a replacement model, alter
the verdict rules, or fabricate evidence when a local dependency fails.

## Capability Probe

Add `scripts/probe_ollama_capabilities.py`, a read-only CLI tool that:

1. Loads the existing application settings.
2. Checks `<OLLAMA_BASE_URL>/api/tags` and verifies the exact configured model
   is installed.
3. Makes a small streamed `/api/chat` request with `think: true`.
4. Parses every NDJSON chunk and counts normal-content and separate
   `message.thinking` chunks.
5. Prints a machine-readable and human-readable result with reachability,
   installed-model status, elapsed time, token metrics, and one of:
   `supported`, `unsupported`, or `probe_failed`.

`supported` means at least one streamed thinking chunk was observed.
`unsupported` means a successful model response had no separate thinking
chunks. `probe_failed` means the host, model, or request failed and capability
is unknown. The command never downloads a model, changes settings, prints, or
persists raw thinking unless a deliberate local `--show-raw-thinking` flag is
used.

## Retry and Local Degradation

Introduce one shared Ollama invocation wrapper used by structured and plain
chat calls. It uses the existing `OLLAMA_MAX_RETRIES` setting as a finite
attempt limit and exponential backoff with jitter for transient transport
failures (connection failure, timeout, overload, or retryable HTTP status).
It must not retry Pydantic validation failures, malformed model output, invalid
configuration, or cancellation. Existing structured-output parse recovery
remains separate and happens after a successful transport response.

Each attempt emits a user-safe progress event. Once the budget is exhausted,
the relevant agent applies its existing local fallback:

- Extractor keeps deterministic sentence splitting and conservative validated
  declarative candidates, marking fallback fidelity.
- Verifier preserves retrieved snippets but returns `INSUFFICIENT_EVIDENCE`,
  zero confidence, and explicit degraded/error processing metadata. It never
  infers truth from model knowledge or missing evidence.
- Reporter renders its deterministic report template, including degraded claim
  counts.
- Dialogue uses its existing clear fallback message.

This is per backend agent action, not unlimited pipeline repetition. A failed
claim must remain isolated so other claims can finish.

## SSE Contract and UI

Extend the existing replayable SSE event hub with additive, backward-compatible
events:

```text
agent_progress {
  agent: "extractor" | "verifier" | "reporter" | "dialogue",
  stage: string,
  status: "started" | "retrying" | "completed" | "degraded",
  attempt?: number,
  max_attempts?: number,
  message: string
}

search_progress {
  claim_index: number,
  query_index: number,
  total_queries: number,
  provider?: string,
  status: "started" | "completed" | "failed",
  result_count?: number
}

thinking_chunk {
  agent: "extractor" | "verifier" | "reporter" | "dialogue",
  stage: string,
  claim_index?: number,
  text: string,
  truncated?: boolean
}
```

`verdict_ready` gains `processing_status` and a short sanitized degraded
reason. The server emits `pipeline_error`, not `pipeline_done`, when the
dialogue graph returns an error result.

The frontend reduces those structured events into a human-readable, collapsible
Activity timeline rather than rendering a flat event log. Agents are the
primary timeline entries, each with a status, duration, and concise summary:

```text
✓ Extractor                 2.1s
  3 claims extracted
● Verifier                  running
  ✓ Claim 1 — verified
  🔎 Claim 2 — searching evidence
○ Reporter                  pending
○ Dialogue                  pending
```

Search operations appear only as nested work below their relevant verifier
claim, with query/provider/result counts available in the expanded detail.
Retries and degradation are translated into short user-facing states such as
`Model unavailable — retrying (2/3)` and `Using degraded verification mode`;
the UI never exposes HTTP codes, Python exception names, stack traces, or raw
SSE field names. A final verifier entry distinguishes `verified`,
`refuted`, `insufficient evidence`, and degraded processing status.

Thinking display is optional and disabled by default. When a capability probe
confirms support and the user enables it for the current run, the browser
receives raw thinking chunks through a dedicated additive `thinking_chunk`
event. The frontend attaches those chunks directly beneath the current
agent/action in the Activity timeline, in quieter text with a live cursor while
streaming:

```text
🔎 Claim 2 — searching evidence
   The retrieved sources need to be compared with the claim… ▌
✓ Claim 1 — verified
```

The thinking region has its own inline collapse control and clear label
`Model-emitted thinking — experimental, not evidence`. Collapsing it hides the
text but retains the relevant activity/result entries. The whole Activity panel
is also collapsible. A per-run size cap retains the newest thinking text and
marks older text as collapsed, so the timeline never becomes an unbounded
transcript. If a model lacks thinking support, no empty thinking region is
shown; the same Activity timeline continues with operation and evidence
summaries only.

Evidence-grounded `ClaimResult.reasoning` remains visually distinct and is the
authoritative explanation of the final verdict. Model-emitted thinking is never
used as evidence or as the source of a verdict.

## Error Handling

- Keep event messages free of prompts, credentials, URLs that failed security
  policy, raw exception stack traces, and raw model text except in opted-in
  thinking chunks.
- Preserve replay ordering for all added events and use terminal error state
  when the pipeline cannot continue.
- Do not suppress a returned dialogue error as a successful completion.
- Continue API compatibility for clients that ignore unknown event types.

## Testing and Quality Gates

- Unit-test retry classification, backoff boundaries, recovery after one
  transient Ollama failure, and exhaustion without retrying validation errors.
- Unit-test capability-probe NDJSON parsing for thinking, content-only,
  missing-model, and network-failure responses.
- Test added SSE event payloads, order/replay, progress after retry, degraded
  `verdict_ready`, and dialogue-error terminal event behavior.
- Add frontend tests for stream state reduction, capability/toggle behavior,
  activity display, and fallback display.
- Correct frontend and backend lint findings, convert keyboard-operable UI
  controls to semantic controls, and update all README status/setup guidance.
- Verify with backend pytest and Ruff plus frontend lint, test, and production
  build.

## Security and Product Constraints

- No cloud-model fallback, automatic `ollama pull`, or hidden model change.
- No truth verdict without retrieved evidence.
- Raw thinking is experimental model output, not a reliable explanation of the
  verdict. The final evidence-grounded reasoning remains the authoritative
  user-facing explanation.
