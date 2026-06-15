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
| `POST` | `/api/dev/extractor/stream` | `{ "input": "string", "metadata": "string \| null" }` | SSE event stream | Dev only |

### Health Response

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model_loaded": true,
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "mistral:7b"
}
```

### SessionDetail

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` | UUID |
| `raw_input` | `string` | Original user input |
| `status` | `string` | `running`, `done`, or `error` |
| `final_report` | `string \| null` | Markdown report from the reporter |
| `error` | `string \| null` | Error message if status is `error` |
| `claim_results` | `list[dict]` | Per-claim verdicts from the verifier |
| `messages` | `list[dict]` | Dialogue message history |
| `created_at` | `float` | Unix timestamp |
| `updated_at` | `float` | Unix timestamp |

### DialogueResponse

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` | UUID |
| `response` | `string` | Dialogue agent reply |
| `intent` | `string` | Classified user intent |
| `needs_new_factcheck` | `bool` | Whether a new fact-check should be triggered |
| `new_claim_text` | `string \| null` | New claim text if a fact-check is needed |
| `error` | `string \| null` | Error message on failure |

### Dev Extractor Stream

`POST /api/dev/extractor/stream` is only registered when `DEV_STREAM_ENABLED=true`. It streams extractor subgraph node updates for local debugging. See [`docs/dev/hack-terminal.md`](../dev/hack-terminal.md).

## SSE Event Types

Events are pushed to the session queue and consumed via `GET /api/sessions/{id}/stream`.

| Event | Payload | Emitted by | Description |
|---|---|---|---|
| `agent_start` | `{ "agent": "string", "timestamp": "ISO8601" }` | Pipeline runner, dialogue runner | An agent begins work (`extractor`, `verifier`, `reporter`, or `dialogue`). |
| `claim_found` | `{ "claim": "string", "index": 0, "total": 1 }` | Pipeline runner | Extractor identifies a claim. |
| `verdict_ready` | `{ "claim": "string", "verdict": "string", "confidence": 0.0, "index": 0, "total": 1 }` | Verifier agent | Verifier completes one claim (emitted per claim during parallel verification). |
| `report_ready` | `{ "final_report": "string" }` | Pipeline runner | Reporter completes the report. |
| `pipeline_done` | `{ "session_id": "string", "duration_seconds": 0.0 }` | Pipeline runner, dialogue runner | Pipeline or dialogue turn completes. |
| `pipeline_error` | `{ "error": "string", "agent": "string" }` | Pipeline runner, dialogue runner | Pipeline or dialogue turn fails visibly. |
| `dialogue_reply` | `{ "message": "string" }` | Dialogue runner | Dialogue agent returns a follow-up answer. |

`search_query` events are not currently emitted at the runner level. Search activity is recorded in each claim result's `search_queries` field.

## Session Lifecycle

1. `POST /api/sessions` creates a session, returns `202` with `status: "running"`, and starts the pipeline in the background.
2. Client opens `GET /api/sessions/{id}/stream` to receive SSE events.
3. On completion, `GET /api/sessions/{id}` returns the full session with `status: "done"`.
4. `POST /api/sessions/{id}/messages` posts a follow-up message (requires `status: "done"`); dialogue runs in the background and emits SSE events on the same stream endpoint.
5. Alternatively, `POST /api/dialogue/{session_id}` runs dialogue synchronously and returns the response directly.
