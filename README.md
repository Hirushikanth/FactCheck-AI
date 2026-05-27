# FactCheck AI

FactCheck AI is a locally deployed, conversational fact-checking system for a final year software engineering project. The system is designed around a LangGraph multi-agent pipeline, a FastAPI backend, and local LLM inference through Ollama.

Phase 1 establishes the runnable foundation only: repository documentation, development toolchain checks, configurable Ollama connectivity, a frozen shared state contract, a minimal health endpoint, and a stub LangGraph pipeline.

## Phase 1 Scope

In scope:

- Architecture documentation and decision records.
- Python 3.11+, Poetry, Node.js 20+, and Git verification.
- Ollama connectivity for `qwen2.5:3b`.
- Schema-first backend scaffold.
- Minimal `GET /api/health` endpoint.

Out of scope for Phase 1:

- Claim extraction logic.
- DuckDuckGo evidence retrieval.
- Verdict generation.
- Reporter and dialogue behavior.
- SQLite persistence.
- Server-Sent Events.
- React frontend implementation.

## Prerequisites

- Python 3.11+
- Poetry 1.8+
- Node.js 20+
- Git
- Ollama with `qwen2.5:3b`

Verify the local toolchain:

```bash
./scripts/verify_toolchain.sh
```

## Ollama Modes

Mode A runs Ollama on this MacBook:

```bash
ollama pull qwen2.5:3b
ollama serve
```

Use:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

Mode B runs Ollama on a Windows PC connected to the same local network. On the Windows host, set `OLLAMA_HOST=0.0.0.0`, allow inbound TCP port `11434`, pull `qwen2.5:3b`, then set the MacBook backend `.env` to:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
```

See [`docs/setup/ollama.md`](docs/setup/ollama.md) for the full setup runbook.

## Backend Quick Start

```bash
cd backend
poetry install
cp .env.example .env
poetry run python ../scripts/smoke_ollama.py
poetry run uvicorn app.main:app --reload
```

From another terminal:

```bash
curl http://localhost:8000/api/health
```

Expected shape:

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model_loaded": true
}
```

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   └── main.py
│   └── factcheck/
│       ├── agents/
│       ├── config.py
│       ├── graph/
│       ├── llm/
│       └── state.py
├── docs/
│   ├── architecture/
│   ├── decisions/
│   └── setup/
└── scripts/
```

## Architecture Reference

- [`docs/architecture/system-overview.md`](docs/architecture/system-overview.md)
- [`docs/architecture/state-schema.md`](docs/architecture/state-schema.md)
- [`docs/architecture/api-and-sse-contract.md`](docs/architecture/api-and-sse-contract.md)
- [`docs/architecture/FactCheckAI_System_Architecture_Design_Document_v1.0.pdf`](docs/architecture/FactCheckAI_System_Architecture_Design_Document_v1.0.pdf)
