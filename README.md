# FactCheck AI

FactCheck AI is a locally deployed, conversational fact-checking system for a final year software engineering project. It accepts natural-language text, decomposes it into atomic claims, retrieves web evidence, and returns evidence-grounded verdicts with confidence scores, explanations, and source URLs.

The backend runs a LangGraph multi-agent pipeline behind a FastAPI API layer, with local LLM inference through Ollama and SQLite session persistence. The shipped React/Vite frontend provides session submission, live SSE pipeline activity, results, and history views.

## Implementation Status

| Component | Status |
|---|---|
| Extractor agent (multi-stage subgraph) | Implemented |
| Verifier agent (parallel per-claim, BM25 ranking, domain credibility tiers) | Implemented |
| Reporter agent | Implemented |
| Dialogue agent (follow-up questions) | Implemented |
| Orchestrator + LangGraph pipeline | Implemented |
| SQLite session persistence | Implemented |
| REST API + SSE streaming | Implemented |
| React TypeScript frontend | Implemented |

## Prerequisites

- Python 3.11+
- Poetry 1.8+
- Git
- Ollama with `gemma4`

Node.js 20+ is checked by `./scripts/verify_toolchain.sh` for the frontend; it is not required to run the backend alone.

Verify the local toolchain:

```bash
./scripts/verify_toolchain.sh
```

## Ollama Modes

Mode A runs Ollama on this MacBook:

```bash
ollama pull gemma4
ollama serve
```

Use:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

Mode B runs Ollama on a Windows PC connected to the same local network. On the Windows host, set `OLLAMA_HOST=0.0.0.0`, allow inbound TCP port `11434`, pull `gemma4`, then set the MacBook backend `.env` to:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
```

See [`docs/setup/ollama.md`](docs/setup/ollama.md) for the full setup runbook.

The original proposal referenced Qwen 2.5 3B; development moved to Mistral 7B for more reliable structured verifier outputs, and the current default is `gemma4`.

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and adjust as needed. All variables are loaded by `AppSettings` in `backend/factcheck/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host (local or LAN) |
| `OLLAMA_MODEL` | `gemma4` | Model name |
| `OLLAMA_TEMPERATURE` | `0.0` | Generation temperature |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `OLLAMA_MAX_RETRIES` | `3` | Retry count |
| `OLLAMA_NUM_CTX` | (blank → Ollama default) | Context window; `8192` recommended for dialogue |
| `OLLAMA_CONCURRENCY` | `1` | Max concurrent Ollama requests |
| `SEARCH_MAX_RESULTS` | `5` | Search result cap per query |
| `SEARCH_PROVIDER_ORDER` | `duckduckgo,tavily,serper` | Provider fallback chain |
| `SEARCH_API_MAX_RETRIES` | `3` | Tavily/Serper total attempts |
| `SEARCH_API_RETRY_BASE_DELAY` | `1.0` | Tavily/Serper backoff base (seconds) |
| `SEARCH_API_RETRY_MAX_DELAY` | `8.0` | Tavily/Serper backoff ceiling (seconds) |
| `SEARCH_API_TIMEOUT_SECONDS` | `30.0` | Tavily/Serper request timeout (seconds) |
| `DDG_MAX_RETRIES` | `3` | DuckDuckGo retry count |
| `DDG_RETRY_BASE_DELAY` | `1.0` | DDG retry backoff base (seconds) |
| `DDG_RETRY_MAX_DELAY` | `8.0` | DDG retry backoff max (seconds) |
| `DDG_MIN_REQUEST_INTERVAL` | `1.5` | DDG minimum spacing between requests |
| `TAVILY_API_KEY` | (empty) | Optional Tavily search API key |
| `SERPER_API_KEY` | (empty) | Optional Serper search API key |
| `FULL_PAGE_FETCH_MODE` | `provider` | Evidence page fetch: `off`, `provider`, or `pinned` |
| `DEMO_REQUIRE_FALLBACK` | `false` | Require a keyed Tavily/Serper fallback at startup |
| `DEV_CORS_ORIGINS` | `http://localhost:5173,...` | CORS allowed origins |
| `SQLITE_PATH` | `factcheck_ai.db` | SQLite database path |
| `DEBUG` | `false` | Debug flag |

DuckDuckGo is used first and does not require credentials. Tavily and Serper are only attempted when keys are configured.

Keep provider keys only in the local `backend/.env` file. That file must never
be committed, pasted into tickets, or included in screenshots/logs; use
`backend/.env.example` only as the key-free template. Set
`DEMO_REQUIRE_FALLBACK=true` for a demonstration run to make startup fail
unless at least one keyed provider (`TAVILY_API_KEY` or `SERPER_API_KEY`) is
included in `SEARCH_PROVIDER_ORDER`. Leave it `false` for deliberate
DuckDuckGo-only development.

### Examiner demonstration checklist

Use this repeatable path immediately before the demonstration:

1. Copy `backend/.env.example` to `backend/.env`, add a real Tavily or Serper
   key locally, and set `DEMO_REQUIRE_FALLBACK=true` with
   `SEARCH_PROVIDER_ORDER=duckduckgo,tavily,serper`.
2. Run the compliance checks:

   ```bash
   cd backend
   poetry run pytest tests/test_search_fallback.py tests/dialogue/test_prompts.py tests/graph/test_thread_isolation.py -q
   poetry run uvicorn app.main:app --reload
   ```

3. Start the frontend with `cd frontend && npm run dev`. For each new browser
   session, the React client creates one UUID with `crypto.randomUUID()` before
   the initial `POST /api/sessions`. The same `session_id` is sent in the
   request, returned unchanged by FastAPI, included in SSE events, and reused
   for every LangGraph fact-check and dialogue invocation.
4. Temporarily disable network access to DuckDuckGo using the local firewall or
   network control, submit a claim, and show the activity timeline/logs moving
   from the DuckDuckGo failure to the keyed Tavily/Serper provider. Restore
   DuckDuckGo access after the demonstration.
5. Explain persistence: `SQLITE_PATH` points to the single SQLite database
   containing the API session tables and LangGraph checkpoint tables. The
   browser session UUID is the LangGraph `thread_id`, so fact-check and
   follow-up dialogue checkpoints remain isolated by session. Do not expose the
   database file or checkpoint contents in the demonstration.

If startup fails while the guard is enabled, check that a non-empty key is
present in `backend/.env` and that its provider name appears in
`SEARCH_PROVIDER_ORDER`.

### Evidence fetch security

When the verifier needs full-page evidence text, the backend prefers Tavily-supplied page content when available. Otherwise it uses a pinned HTTP fetch that validates each URL, resolves DNS to a public IP, connects to that IP directly, and re-validates every redirect hop. Private, loopback, link-local, and metadata targets are blocked. Before LAN or public deployment, also restrict outbound network access at the infrastructure layer (for example, deny RFC1918 and `169.254.0.0/16` egress from the backend host).

## Backend Quick Start

```bash
cd backend
poetry install
cp .env.example .env
poetry run python ../scripts/smoke_ollama.py
poetry run uvicorn app.main:app --reload
```

The server runs at `http://localhost:8000`.

## Frontend Quick Start

With the backend running, start the React/Vite frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to the backend at
`http://localhost:8000`; update the proxy in `frontend/vite.config.ts` if the
backend runs elsewhere.

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
  "ollama_model": "gemma4"
}
```

Create a session and start the pipeline (returns `202 Accepted`):

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"input": "The Earth is round.", "session_id": "11111111-1111-4111-8111-111111111111"}'
```

The browser-generated UUID is authoritative; the placeholder above is only
for a manual API smoke test.

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

Run the frontend quality checks:

```bash
cd frontend
npm run test
npm run lint
npm run build
```

Run optional Ollama-backed integration tests (requires a running Ollama instance):

```bash
RUN_OLLAMA_INTEGRATION=1 poetry run pytest -m integration
```

## Dev Console

A temporary local UI for testing the full session flow (claim → SSE → report → follow-up). The `dev-console/` folder is gitignored and kept only on your machine for development.

```bash
# Terminal 1 — backend
cd backend && poetry run uvicorn app.main:app --reload

# Terminal 2 — dev console (local folder, not in git)
cd dev-console && python3 -m http.server 8080
```

Open `http://localhost:8080`. Ensure `DEV_CORS_ORIGINS` in `backend/.env` includes `http://localhost:8080`.

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point (v0.6.0)
│   │   ├── routers/                 # sessions, dialogue
│   │   └── schemas/                 # Pydantic API models
│   ├── scripts/                     # stress tests (extractor, verifier)
│   └── factcheck/
│       ├── agents/                  # orchestrator, extractor, verifier, reporter
│       ├── config.py                # AppSettings from .env
│       ├── db/                      # SQLite session store
│       ├── dialogue/                # follow-up dialogue graph
│       ├── extractor/               # multi-stage claim extractor subgraph
│       ├── graph/                   # pipeline runner + SSE event hub
│       ├── llm/                     # Ollama factory + structured output
│       ├── reporter/                # report generation
│       ├── search/                  # DuckDuckGo → Tavily → Serper fallback
│       ├── state.py                 # shared FactCheckState schema
│       ├── streaming/               # SSE event formatting
│       └── verifier/                # evidence retrieval + evaluation
├── docs/
│   ├── architecture/              # system overview, state schema, API contract
│   ├── decisions/                 # ADRs
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
