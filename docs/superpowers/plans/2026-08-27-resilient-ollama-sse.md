# Resilient Ollama and SSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver observable, retry-resilient local Ollama fact-check runs with a collapsible agent-first frontend timeline and optional raw model-thinking stream.

**Architecture:** A tested Ollama transport wrapper centralizes finite retry classification and progress callbacks while retaining existing structured-output recovery. The backend publishes additive replayable SSE events for agent/search progress and thinking chunks; the React client reduces them into a timeline whose verifier children contain search, verdict, retry, fallback, and optional collapsible thinking details.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, httpx, Ollama, pytest; React 19, TypeScript, Vite, TanStack Query, Vitest, React Testing Library.

**Spec:** `docs/superpowers/specs/2026-08-27-resilient-ollama-sse-design.md`

## Global Constraints

- Do not add a cloud-model fallback, automatic `ollama pull`, or hidden model change.
- Never produce a truth verdict without retrieved evidence; degraded verifier fallback is `INSUFFICIENT_EVIDENCE` at confidence `0.0`.
- Retry only transient Ollama transport failures; do not retry cancellation, invalid configuration, malformed model output, or Pydantic validation failures.
- All new SSE events are additive, replayable, ordered, and contain only user-safe messages—no credentials, stack traces, HTTP codes, or prompts.
- Raw model thinking is opt-in for a single run, probe-gated, size-capped, and labeled `Model-emitted thinking — experimental, not evidence`.
- Frontend renders the event model as a collapsible agent-first timeline, never a flat log or raw SSE payloads.
- New behavior follows TDD: record a failing focused test before its production implementation.

---

## File Structure

- `backend/factcheck/llm/retry.py`: retryable failure classification, bounded backoff, and injectable retry callback.
- `backend/factcheck/llm/ollama.py`: direct read-only capability probe helpers and shared transport wiring.
- `scripts/probe_ollama_capabilities.py`: CLI diagnostic for configured Ollama and thinking support.
- `backend/factcheck/graph/event_bus.py`, `runner.py`, agent/subgraph nodes: additive progress and thinking events.
- `backend/tests/llm/test_retry.py`, `tests/test_ollama_probe.py`, `tests/graph/test_runner.py`: resilience/SSE regression coverage.
- `frontend/src/activity/types.ts`, `activity/reducer.ts`: independently testable timeline state model and SSE-to-UI reducer.
- `frontend/src/components/ActivityTimeline.tsx`: accessible collapsible timeline and inline thinking view.
- `frontend/src/hooks/useSessionStream.ts`, `api/types.ts`, `screens/SessionScreen.tsx`: stream integration.
- `frontend/src/**/*.test.ts(x)`: Vitest coverage for reducer, component behavior, and stream parsing.
- Root and frontend READMEs: accurate shipped frontend, diagnostic command, and UI behavior.

### Task 1: Establish frontend test tooling and clean baseline quality

**Files:**
- Modify: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/src/App.tsx`, `frontend/src/hooks/useSessionStream.ts`
- Create: `frontend/src/test/setup.ts`, `frontend/src/lib/verdict.test.ts`, `frontend/src/api/client.test.ts`
- Modify: `README.md`, `frontend/README.md`

**Interfaces:**
- Produces: `npm run test` runs Vitest in jsdom; existing frontend source passes `npm run lint` and `npm run build`.
- Consumes: existing Vite proxy and API client exports.

- [ ] **Step 1: Add failing frontend test commands and test files**

Add `test: "vitest run"` and `test:watch: "vitest"`; write a verdict-label test and a mocked-fetch API error test before adding Vitest configuration. Run `npm run test` and confirm it fails because `vitest` is unavailable.

- [ ] **Step 2: Install and configure test dependencies**

Add `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`, and `@testing-library/user-event` as dev dependencies. Configure `test.environment = "jsdom"` and `test.setupFiles = "./src/test/setup.ts"` in `vite.config.ts`; import `@testing-library/jest-dom/vitest` from setup. Run `npm run test` and confirm both tests pass.

- [ ] **Step 3: Correct existing lint violations with regression coverage**

Move `AppContext`, `AppTab`, and `useApp` from `src/App.tsx` into `src/app-context.ts`; update screen imports. Remove the useless `data` assignment in the SSE parser while preserving JSON parsing. Run `npm run lint` and confirm zero errors, then `npm run build`.

- [ ] **Step 4: Replace stale documentation**

Update root README implementation status and quick start to describe the shipped React frontend (`cd frontend && npm install && npm run dev`). Replace the Vite template frontend README with application-specific setup, test, build, proxy, health, and activity/timeline notes. Confirm no `planned frontend` text remains with `rg -n "planned.*frontend|frontend.*planned" README.md frontend/README.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md frontend
git commit -m "test: establish frontend quality baseline"
```

### Task 2: Add test-first local Ollama retry and capability probe

**Files:**
- Create: `backend/factcheck/llm/retry.py`, `backend/tests/llm/test_retry.py`, `backend/tests/test_ollama_probe.py`, `scripts/probe_ollama_capabilities.py`
- Modify: `backend/factcheck/llm/ollama.py`, `backend/factcheck/llm/structured.py`, `backend/factcheck/llm/extractor_structured.py`, `backend/factcheck/config.py`, `backend/.env.example`

**Interfaces:**
- Produces: `invoke_with_ollama_retry(operation, *, max_attempts, on_retry)` and `probe_ollama_capabilities(settings, client)`.
- Produces: probe statuses exactly `supported`, `unsupported`, `probe_failed`.
- Consumes: `AppSettings.ollama_max_retries`, existing process-wide Ollama semaphore, and `httpx` transports in tests.

- [ ] **Step 1: Write failing retry tests**

Test an operation that raises `httpx.ConnectError` once then succeeds; assert two attempts and a callback payload `{attempt: 2, max_attempts: 3}`. Test exhaustion after three transient failures. Test that `pydantic.ValidationError`, `asyncio.CancelledError`, and `ValueError` each make exactly one attempt. Run `poetry run pytest tests/llm/test_retry.py -v` and confirm failing import/behavior.

- [ ] **Step 2: Implement minimal retry wrapper**

Implement retryable classification for `httpx.TimeoutException`, `httpx.NetworkError`, and explicit retryable status errors only. Use capped exponential backoff with jitter and injectable sleep/random functions for deterministic tests. Call `on_retry` only before a subsequent attempt. Run the focused tests until green.

- [ ] **Step 3: Write and run failing probe parsing tests**

Model a streamed NDJSON response with `message.content`, a response with `message.thinking`, a missing model tags payload, and connection failure. Assert exact status and that raw thinking is omitted unless `show_raw_thinking=True`. Run `poetry run pytest tests/test_ollama_probe.py -v` and verify the missing helper/test failure.

- [ ] **Step 4: Implement the read-only probe and CLI**

Use direct `httpx.AsyncClient` against `/api/tags` then streamed `/api/chat` with `{model, messages, stream: true, think: true}`. Parse line-delimited JSON; collect chunk counts, final duration/token fields, and optional raw thinking. Provide `--json` and `--show-raw-thinking`; never write config or fetch a model. Return nonzero only for `probe_failed`. Run probe tests and `python ../scripts/probe_ollama_capabilities.py --help`.

- [ ] **Step 5: Integrate retry without changing structured recovery semantics**

Wrap only each remote `ainvoke` transport invocation in both structured helper modules; retain existing JSON/schema fallback order. Preserve semaphore use around each attempt. Add tests proving the existing schema fallback is not retried as a transport error. Run focused LLM tests plus `poetry run pytest tests/test_llm_structured.py tests/test_extractor_structured.py -v`.

- [ ] **Step 6: Commit**

```bash
git add backend scripts/probe_ollama_capabilities.py
git commit -m "feat: add Ollama retry and capability probe"
```

### Task 3: Publish replayable SSE resilience and thinking events

**Files:**
- Modify: `backend/factcheck/graph/event_bus.py`, `backend/factcheck/graph/runner.py`, `backend/factcheck/agents/extractor.py`, `backend/factcheck/agents/verifier.py`, `backend/factcheck/verifier/nodes/retriever.py`, `backend/factcheck/dialogue/__init__.py`
- Modify: `backend/tests/graph/test_event_bus.py`, `backend/tests/graph/test_runner.py`, `backend/tests/agents/test_verifier_agent.py`, `backend/tests/dialogue/test_dialogue_route.py`

**Interfaces:**
- Produces: ordered `agent_progress`, `search_progress`, and gated `thinking_chunk` stream events defined in the spec.
- Produces: `verdict_ready` additions `processing_status` and `degraded_reason`.
- Consumes: current replay hub `publish()`/`stream_events()` and `run_factcheck_with_events()`.

- [ ] **Step 1: Write failing event contract tests**

Add tests that subscribe/replay a hub and assert the exact payload fields/order for `agent_progress`, `search_progress`, and `thinking_chunk`. Add a verifier event test that asserts error/degraded status is present and does not expose raw `processing_error`. Add a dialogue runner test where `run_dialogue()` returns `{"error": "..."}` and assert `pipeline_error` appears without `pipeline_done`. Run focused graph/dialogue tests and verify failures.

- [ ] **Step 2: Implement event payload constructors and terminal-state correction**

Centralize event payload validation/sanitization near runner/event bus. Emit stage starts/completions/degradation, nested search progress, and sanitized verdict state. Convert a returned dialogue error into `pipeline_error` and persist an error status. Run focused tests to green.

- [ ] **Step 3: Connect retry callbacks to progress events**

Pass agent/stage context to the retry wrapper at every LLM call path. Emit message exactly `Model unavailable — retrying (N/M)` and never include exception strings. Ensure retries do not create duplicate terminal events. Add a test for retry progress followed by successful stage completion.

- [ ] **Step 4: Gate and emit thinking chunks**

Add an explicit per-run option, default false, that is rejected/ignored unless probe-confirmed capability is available. Stream direct Ollama `thinking` chunks through the additive event only when opted in; cap total text and set `truncated: true` on the final cap event. Preserve existing LangChain response behavior for non-thinking calls. Add tests for default suppression, enabled forwarding, and cap behavior.

- [ ] **Step 5: Commit**

```bash
git add backend
git commit -m "feat: stream resilient pipeline activity"
```

### Task 4: Build the accessible Activity timeline and thinking reducer

**Files:**
- Create: `frontend/src/activity/types.ts`, `frontend/src/activity/reducer.ts`, `frontend/src/activity/reducer.test.ts`, `frontend/src/components/ActivityTimeline.tsx`, `frontend/src/components/ActivityTimeline.test.tsx`
- Modify: `frontend/src/api/types.ts`, `frontend/src/hooks/useSessionStream.ts`, `frontend/src/screens/SessionScreen.tsx`, `frontend/src/styles/session.css`

**Interfaces:**
- Produces: `reduceActivityEvent(state, event)` with agent primary entries, verifier claim/search children, durations, retry/degraded messages, and per-action thinking buffers.
- Produces: `ActivityTimeline({ timeline, thinkingEnabled, onThinkingEnabledChange })` with keyboard-operable collapsible controls.
- Consumes: SSE event payload types from Task 3 and existing session stream callbacks.

- [ ] **Step 1: Write failing reducer tests**

Test agent start/completion duration, nested verifier search state, translated retry message, degraded verdict result, ignored unknown events, and thinking chunks attached to the current verifier action. Test text truncation preserves newest chunks and marks an entry truncated. Run `npm run test -- activity/reducer.test.ts` and verify failure because reducer is missing.

- [ ] **Step 2: Implement pure timeline types and reducer**

Define discriminated frontend event types matching the backend contract. Implement immutable pure state transitions; use a monotonic clock injection for durations; retain agent order Extractor, Verifier, Reporter, Dialogue. Run reducer tests to green.

- [ ] **Step 3: Write failing accessible component tests**

Render active verifier state and assert agent hierarchy, nested claim text, and human-friendly retry copy. Use `userEvent.keyboard` to collapse/expand the Activity timeline and the inline thinking region. Assert raw technical strings and empty thinking regions are absent. Run the focused test and observe component/import failure.

- [ ] **Step 4: Implement timeline component and integrate stream state**

Render semantic `<section>`, headings, buttons with `aria-expanded`, and lists for hierarchy. Render thinking in subdued text with `aria-live="polite"` only while active; display the experimental/not-evidence label and live cursor. Update `useSessionStream` to parse new events and reduce activity state; render the timeline in SessionScreen. Replace clickable non-semantic session/history controls touched by this flow with buttons or equivalent keyboard handlers. Run component, hook, lint, and build checks.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat: add live fact-check activity timeline"
```

### Task 5: End-to-end contract, visual verification, and final cleanup

**Files:**
- Modify: `docs/architecture/api-and-sse-contract.md`, `README.md`, `frontend/README.md`
- Create or modify: backend/frontend integration tests required by test setup
- Modify: all files reported by `poetry run ruff check .` until clean

**Interfaces:**
- Consumes: probe, retry, SSE, and timeline behavior from Tasks 1–4.
- Produces: documented client contract and clean backend/frontend quality gates.

- [ ] **Step 1: Write end-to-end stream contract test**

Construct a fake fact-check run that emits agent start, retry, search progress, verdict, reporter complete, and pipeline done. Assert the HTTP SSE sequence can be replayed and frontend reducer produces the expected final timeline. Run the test and confirm it fails before any fixture/support code is added.

- [ ] **Step 2: Implement only the required test fixture/support**

Use existing test app/event hub seams, not a real Ollama process. Keep the test deterministic and verify a degraded verifier still permits a deterministic report. Run backend integration and frontend test suites.

- [ ] **Step 3: Finish lint and documentation contract**

Resolve all reported Ruff violations without broad refactors. Document all new event names/payloads, thinking opt-in behavior, capability probe commands/statuses, retry/fallback semantics, and the timeline hierarchy. Run `poetry run ruff check .` and verify no findings.

- [ ] **Step 4: Visually validate the frontend**

Run backend with a deterministic test/fake stream or local Ollama when reachable, start the Vite app, and inspect the Session screen in a browser at desktop and narrow widths. Verify: timeline nesting, collapse controls, retry/degraded copy, no raw errors, live-thinking label/cursor when enabled, and clean unsupported-model state. Capture screenshots or describe any environment-blocked live-model limitation in the final report.

- [ ] **Step 5: Run full verification**

```bash
cd backend && poetry run pytest && poetry run ruff check .
cd ../frontend && npm run test && npm run lint && npm run build
```

- [ ] **Step 6: Commit**

```bash
git add README.md docs backend frontend
git commit -m "test: verify resilient fact-check experience"
```
