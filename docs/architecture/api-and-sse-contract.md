# API and SSE Contract

This document records the implemented REST and Server-Sent Events contract for the FactCheck AI backend (v0.6.0).

## REST Endpoints

| Method | Path | Request Body | Response | Status |
|---|---|---|---|---|
| `GET` | `/api/health` | none | Health payload (see below) | Implemented |
| `POST` | `/api/sessions` | `{ "input": "string" }` | `{ "session_id": "string", "status": "string" }` | Implemented (202) |
| `GET` | `/api/sessions/{id}/stream` | none | SSE event stream | Implemented |
| `GET` | `/api/sessions/{id}` | none | `SessionDetail` | Implemented |
| `POST` | `/api/sessions/{id}/messages` | `{ "message": "string" }` | `{ "message_id": "string" }` | Implemented (202) |
| `GET` | `/api/sessions` | none | `list[SessionSummary]` | Implemented |
| `DELETE` | `/api/sessions/{id}` | none | `{ "deleted": true }` | Implemented |
| `POST` | `/api/dialogue/{session_id}` | `{ "message": "string" }` | `DialogueResponse` | Implemented |

### Health Response

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model_loaded": true,
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "gemma4"
}
```

### SessionDetail

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` | UUID |
| `active_run_id` | `string \| null` | UUID of the currently active fact-check run |
| `raw_input` | `string` | Input text for the **active** run |
| `status` | `string` | `running`, `done`, or `error` |
| `final_report` | `string \| null` | Markdown report from the reporter (active run) |
| `error` | `string \| null` | Error message if status is `error` |
| `claim_results` | `list[dict]` | Per-claim verdicts from the verifier (active run) |
| `messages` | `list[dict]` | Dialogue message history |
| `runs` | `list[FactCheckRunSummary]` | All fact-check runs in sequence order |
| `created_at` | `float` | Unix timestamp |
| `updated_at` | `float` | Unix timestamp |

### FactCheckRunSummary

| Field | Type | Description |
|---|---|---|
| `run_id` | `string` | UUID |
| `sequence` | `int` | 1-based order within the session |
| `raw_input` | `string` | Input text for this run |
| `status` | `string` | `running`, `done`, or `error` |
| `triggered_by` | `string` | `initial` or `dialogue` |
| `created_at` | `float` | Unix timestamp |

### DialogueResponse

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` | UUID |
| `response` | `string` | Dialogue agent reply |
| `intent` | `string` | Classified user intent |
| `needs_new_factcheck` | `bool` | Whether a new fact-check should be triggered |
| `new_claim_text` | `string \| null` | New claim text if a fact-check is needed |
| `error` | `string \| null` | Error message on failure |

## SSE Event Types

Events are pushed to the per-session event hub and consumed via `GET /api/sessions/{id}/stream`. The hub buffers events (ring buffer, max 256) and replays them to new subscribers while a run is active or for up to 120 seconds after close.

| Event | Payload | Emitted by | Description |
|---|---|---|---|
| `stream_open` | `{ "session_id": "string", "run_id": "string \| null", "replay_count": 0, "hub_state": "open \| closed", "server_time": "ISO8601" }` | Event hub | First frame on every successful stream connection; confirms the stream is live. |
| `agent_start` | `{ "agent": "string", "timestamp": "ISO8601" }` | Pipeline runner, dialogue runner | An agent begins work (`extractor`, `verifier`, `reporter`, or `dialogue`). |
| `claim_found` | `{ "claim": "string", "index": 0, "total": 1 }` | Pipeline runner | Extractor identifies a claim. |
| `extractor_stage_failed` | `{ "stage": "string", "sentence": "string", "reason": "string", "successes": 0, "attempts": 0, "timestamp": "ISO8601" }` | Extractor agent | A sentence was dropped during selection, disambiguation, or decomposition. |
| `verdict_ready` | `{ "claim": "string", "verdict": "string", "confidence": 0.0, "index": 0, "total": 1 }` | Verifier agent | Verifier completes one claim (emitted per claim during parallel verification). |
| `report_ready` | `{ "final_report": "string" }` | Pipeline runner | Reporter completes the report. |
| `pipeline_done` | `{ "session_id": "string", "duration_seconds": 0.0 }` | Pipeline runner, dialogue runner | Pipeline or dialogue turn completes. |
| `pipeline_error` | `{ "error": "string", "agent": "string" }` | Pipeline runner, dialogue runner | Pipeline or dialogue turn fails visibly. |
| `dialogue_reply` | `{ "message": "string" }` | Dialogue runner | Dialogue agent returns a follow-up answer. |

`search_query` events are not currently emitted at the runner level. Search activity is recorded in each claim result's `search_queries` field.

### Stream Endpoint Errors

| Status | When | Response |
|---|---|---|
| `404` | Unknown `session_id` | `{ "detail": "Session not found" }` |
| `409` | No subscribable hub (missed window or pipeline orphaned) | `{ "detail": { "code": "stream_missed" \| "pipeline_orphaned", "session_status": "string", "active_run_id": "string \| null", "hint": "string" } }` |
| `200` | Active or replayable hub | SSE stream (first event is always `stream_open`) |

**409 codes:**

- `stream_missed` — session is `done` or `error` and no hub is available (client connected too late; use `GET /api/sessions/{id}` for final state).
- `pipeline_orphaned` — session is `running` but no hub appeared within 5 seconds (unexpected; retry or fetch session state).

### Client Reconnect Rules

1. Open `GET /api/sessions/{id}/stream` immediately after every `202` (`POST /api/sessions`, `POST /api/sessions/{id}/messages`).
2. Treat HTTP `200` alone as insufficient — wait for the `stream_open` event before assuming the stream is valid.
3. On `409` with `code: "stream_missed"` and `session_status: "done"`, use `GET /api/sessions/{id}` for authoritative results.
4. After receiving `pipeline_done`, if `GET /api/sessions/{id}` shows `status: "running"` (e.g. dialogue triggered a new fact-check), reconnect to `/stream`.
5. On disconnect mid-run, reconnect once; buffered events are replayed while the hub is open or within the 120-second post-close TTL.

## Session Lifecycle

1. `POST /api/sessions` creates a session and **run #1** (`triggered_by: initial`), returns `202` with `status: "running"`, and starts the pipeline in the background.
2. Client opens `GET /api/sessions/{id}/stream` to receive SSE events.
3. On completion, `GET /api/sessions/{id}` returns the full session with `status: "done"`. Top-level `raw_input`, `claim_results`, and `final_report` reflect the **active run**; `runs[]` lists all runs in order.
4. `POST /api/sessions/{id}/messages` posts a follow-up message (requires `status: "done"`); dialogue runs in the background and emits SSE events on the same stream endpoint.
5. If dialogue detects a new claim, a **new run** is appended (`triggered_by: dialogue`), becomes the active run, and prior runs are preserved in `runs[]`.
6. Alternatively, `POST /api/dialogue/{session_id}` runs dialogue synchronously and returns the response directly.

`GET /api/sessions` (list) uses the **first run's** `raw_input` as the session title.
