# FactCheck AI

FactCheck AI is a locally deployed, conversational fact-checking system for a final year software engineering project. The system is designed around a LangGraph multi-agent pipeline, a FastAPI backend, and local LLM inference through Ollama.

Phase 2 implements the extractor agent and prepares evidence-search fallback infrastructure. The extractor uses a Claimify-style subgraph to turn raw text into atomic, verifiable claims, while search is available as a reusable DuckDuckGo → Tavily → Serper fallback layer for the future verifier phase.

## Phase 2 Scope

In scope:

- Claimify-style extractor subgraph: sentence splitting, selection, disambiguation, decomposition, and validation.
- Main LangGraph integration that writes extracted claims into `FactCheckState`.
- Search fallback module with DuckDuckGo primary and optional Tavily / Serper API fallbacks.
- Tests for extractor nodes, graph wiring, search fallback behavior, and settings.

Out of scope for Phase 2:

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
- Ollama with `mistral:7b`

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

## Search Fallback Configuration

DuckDuckGo is used first and does not require credentials. Tavily and Serper are only attempted when keys are configured:

```bash
SEARCH_MAX_RESULTS=5
SEARCH_PROVIDER_ORDER=duckduckgo,tavily,serper
TAVILY_API_KEY=
SERPER_API_KEY=
```

## Backend Quick Start

```bash
cd backend
poetry install
cp .env.example .env
poetry run python ../scripts/smoke_ollama.py
poetry run uvicorn app.main:app --reload
```

Run the extractor tests:

```bash
poetry run pytest tests/test_extractor_utils.py tests/test_extractor_nodes.py tests/test_extractor_graph.py tests/test_extractor_agent.py
```

Run the optional Ollama-backed extractor integration test:

```bash
RUN_OLLAMA_INTEGRATION=1 poetry run pytest tests/test_ollama_extractor_integration.py
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
│       ├── extractor/
│       ├── graph/
│       ├── llm/
│       ├── search/
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
