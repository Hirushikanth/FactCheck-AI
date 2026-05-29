# Hacker Terminal Dev SSE

The Hacker Terminal is a temporary local UI for watching the extractor LangGraph subgraph stream node updates. It is intentionally not the production frontend and the generated `dev-hack-terminal/` folder is ignored by git.

## Enable The Dev Stream

Set this in `backend/.env`:

```bash
DEV_STREAM_ENABLED=true
```

Then run the backend:

```bash
cd backend
poetry run uvicorn app.main:app --reload
```

The dev-only endpoint is:

```text
POST http://127.0.0.1:8000/api/dev/extractor/stream
```

Example request:

```bash
curl -N \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/api/dev/extractor/stream \
  -d '{"input":"The Earth is round. Water boils at 100 degrees Celsius at sea level."}'
```

## Run The Temporary UI

Create or use the local ignored folder:

```bash
cd dev-hack-terminal
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080
```

The UI posts JSON to the backend and parses the `text/event-stream` response with `fetch`, because `EventSource` cannot send a request body.
