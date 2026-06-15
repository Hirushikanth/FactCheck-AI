# FactCheck AI

FactCheck AI is a locally deployed, conversational fact-checking system for a final year software engineering project. It accepts natural-language text, decomposes it into atomic claims, retrieves web evidence, and returns evidence-grounded verdicts with confidence scores, explanations, and source URLs.

The backend runs a LangGraph multi-agent pipeline behind a FastAPI API layer, with local LLM inference through Ollama and SQLite session persistence. A production React frontend is planned but not yet implemented.

## Implementation Status

| Component | Status |
|---|---|
| Extractor agent (Claimify-style subgraph) | Implemented |
| Verifier agent (parallel per-claim, BM25 ranking, domain credibility tiers) | Implemented |
| Reporter agent | Implemented |
| Dialogue agent (follow-up questions) | Implemented |
| Orchestrator + LangGraph pipeline | Implemented |
| SQLite session persistence | Implemented |
| REST API + SSE streaming | Implemented |
| React TypeScript frontend | Planned |

## Prerequisites

- Python 3.11+
- Poetry 1.8+
- Git
- Ollama with `mistral:7b`

Node.js 20+ is checked by `./scripts/verify_toolchain.sh` for the planned frontend; it is not required to run the backend today.

Verify the local toolchain:

```bash
./scripts/verify_toolchain.sh
```

## Ollama Modes

Mode A runs Ollama on this MacBook:

```bash
ollama pull mistral:7b
ollama serve
```

Use:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

Mode B runs Ollama on a Windows PC connected to the same local network. On the Windows host, set `OLLAMA_HOST=0.0.0.0`, allow inbound TCP port `11434`, pull `mistral:7b`, then set the MacBook backend `.env` to:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
```

See [`docs/setup/ollama.md`](docs/setup/ollama.md) for the full setup runbook.

The original proposal referenced Qwen 2.5 3B, but the implementation uses Mistral 7B because it produced more reliable structured verifier outputs during development.

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and adjust as needed. All variables are loaded by `AppSettings` in `backend/factcheck/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host (local or LAN) |
| `OLLAMA_MODEL` | `mistral:7b` | Model name |
| `OLLAMA_TEMPERATURE` | `0.0` | Generation temperature |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `OLLAMA_MAX_RETRIES` | `3` | Retry count |
| `OLLAMA_NUM_CTX` | (blank → Ollama default) | Context window; `8192` recommended for dialogue |
| `OLLAMA_CONCURRENCY` | `1` | Max concurrent Ollama requests |
| `SEARCH_MAX_RESULTS` | `5` | Search result cap per query |
| `SEARCH_PROVIDER_ORDER` | `duckduckgo,tavily,serper` | Provider fallback chain |
| `DDG_MAX_RETRIES` | `3` | DuckDuckGo retry count |
| `DDG_RETRY_BASE_DELAY` | `1.0` | DDG retry backoff base (seconds) |
| `DDG_RETRY_MAX_DELAY` | `8.0` | DDG retry backoff max (seconds) |
| `DDG_MIN_REQUEST_INTERVAL` | `1.5` | DDG minimum spacing between requests |
| `TAVILY_API_KEY` | (empty) | Optional Tavily search API key |
| `SERPER_API_KEY` | (empty) | Optional Serper search API key |
| `DEV_STREAM_ENABLED` | `false` | Enable dev extractor SSE endpoint |
| `DEV_CORS_ORIGINS` | `http://localhost:5173,...` | CORS allowed origins |
| `SQLITE_PATH` | `factcheck_ai.db` | SQLite database path |
| `DEBUG` | `false` | Debug flag |

DuckDuckGo is used first and does not require credentials. Tavily and Serper are only attempted when keys are configured.

## Backend Quick Start

```bash
cd backend
poetry install
cp .env.example .env
poetry run python ../scripts/smoke_ollama.py
poetry run uvicorn app.main:app --reload
```

The server runs at `http://localhost:8000`.

## API Usage

Health check:

```bash
curl http://localhost:8000/api/health
```

Expected shape:

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model_loaded": true,
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "mistral:7b"
}
```

Create a session and start the pipeline (returns `202 Accepted`):

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"input": "The Earth is round."}'
```

Stream SSE progress (use `session_id` from the response above):

```bash
curl -N http://localhost:8000/api/sessions/{session_id}/stream
```

Retrieve completed session state:

```bash
curl http://localhost:8000/api/sessions/{session_id}
```

Post a follow-up message after the pipeline completes (SSE via the same stream endpoint):

```bash
curl -X POST http://localhost:8000/api/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "What sources did you use?"}'
```

Synchronous dialogue (no SSE):

```bash
curl -X POST http://localhost:8000/api/dialogue/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"message": "What sources did you use?"}'
```

List or delete sessions:

```bash
curl http://localhost:8000/api/sessions
curl -X DELETE http://localhost:8000/api/sessions/{session_id}
```

See [`docs/architecture/api-and-sse-contract.md`](docs/architecture/api-and-sse-contract.md) for the full REST and SSE contract.

## Tests

Run the full test suite:

```bash
cd backend
poetry run pytest
```

Run optional Ollama-backed integration tests (requires a running Ollama instance):

```bash
RUN_OLLAMA_INTEGRATION=1 poetry run pytest -m integration
```

## Dev Hack Terminal

A temporary local UI for watching extractor subgraph SSE output is available when `DEV_STREAM_ENABLED=true`. See [`docs/dev/hack-terminal.md`](docs/dev/hack-terminal.md).

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point (v0.6.0)
│   │   └── routers/                 # sessions, dialogue, dev_stream
│   └── factcheck/
│       ├── agents/                  # orchestrator, extractor, verifier, reporter
│       ├── config.py                # AppSettings from .env
│       ├── db/                      # SQLite session store
│       ├── dialogue/                # follow-up dialogue graph
│       ├── extractor/               # Claimify-style subgraph
│       ├── graph/                   # pipeline runner + SSE event bus
│       ├── llm/                     # Ollama factory + structured output
│       ├── reporter/                # report generation
│       ├── search/                  # DuckDuckGo → Tavily → Serper fallback
│       ├── state.py                 # shared FactCheckState schema
│       └── verifier/                # evidence retrieval + evaluation
├── docs/
│   ├── architecture/              # system overview, state schema, API contract
│   ├── decisions/                 # ADRs
│   ├── dev/                       # hack-terminal docs
│   └── setup/                     # Ollama runbook
└── scripts/
    ├── smoke_ollama.py
    └── verify_toolchain.sh
```

## Architecture Reference

- [`docs/architecture/system-overview.md`](docs/architecture/system-overview.md)
- [`docs/architecture/state-schema.md`](docs/architecture/state-schema.md)
- [`docs/architecture/api-and-sse-contract.md`](docs/architecture/api-and-sse-contract.md)
- [`docs/architecture/FactCheckAI_System_Architecture_Design_Document_v1.0.pdf`](docs/architecture/FactCheckAI_System_Architecture_Design_Document_v1.0.pdf)
