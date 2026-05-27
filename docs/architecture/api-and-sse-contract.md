# API and SSE Contract

This document records the planned API and Server-Sent Events contract. In Phase 1, only `GET /api/health` is implemented. Session endpoints, persistence, and streaming are implemented in Phase 6.

## REST Endpoints

| Method | Path | Request Body | Response | Phase |
|---|---|---|---|---|
| `POST` | `/api/sessions` | `{ "input": "string" }` | `{ "session_id": "string", "status": "string" }` | Phase 6 |
| `GET` | `/api/sessions/{id}/stream` | none | SSE event stream | Phase 6 |
| `GET` | `/api/sessions/{id}` | none | `FactCheckSession` | Phase 6 |
| `POST` | `/api/sessions/{id}/messages` | `{ "message": "string" }` | `{ "message_id": "string" }` | Phase 6 |
| `GET` | `/api/sessions` | none | `list[SessionSummary]` | Phase 6 |
| `DELETE` | `/api/sessions/{id}` | none | `{ "deleted": true }` | Phase 6 |
| `GET` | `/api/health` | none | `{ "status": "ok", "ollama_reachable": true, "model_loaded": true }` | Phase 1 |

## SSE Event Types

| Event | Payload | Description |
|---|---|---|
| `agent_start` | `{ "agent": "string", "timestamp": "ISO8601" }` | Orchestrator activates an agent. |
| `claim_found` | `{ "claim": "string", "index": 0, "total": 1 }` | Extractor identifies a claim. |
| `search_query` | `{ "query": "string", "claim_index": 0 }` | Verifier issues a search query. |
| `verdict_ready` | `{ "claim": "string", "verdict": "string", "confidence": 0.0 }` | Verifier completes a claim result. |
| `report_ready` | `{ "final_report": "string" }` | Reporter completes the report. |
| `pipeline_done` | `{ "session_id": "string", "duration_seconds": 0.0 }` | Pipeline completes successfully. |
| `pipeline_error` | `{ "error": "string", "agent": "string" }` | Pipeline fails visibly. |
| `dialogue_reply` | `{ "message": "string" }` | Dialogue agent returns a follow-up answer. |
