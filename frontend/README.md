# FactCheck AI frontend

This React and TypeScript application is the user interface for FactCheck AI.
It submits claims to the FastAPI backend, renders replayable Server-Sent Events
while the multi-agent pipeline runs, and provides results and session history.

## Development

Install dependencies and start Vite from this directory:

```bash
npm install
npm run dev
```

The app is served at `http://localhost:5173`. During development, Vite proxies
requests under `/api` to `http://localhost:8000`; start the backend separately
with its documented Poetry setup. Change the target in `vite.config.ts` when
using another backend host.

## Quality checks

```bash
npm run test       # Vitest tests in jsdom
npm run lint       # ESLint
npm run build      # TypeScript check and production bundle
```

The session screen displays pipeline activity as it arrives over SSE. The
Results and History tabs can be used to inspect completed sessions. The backend
health indicator in the top bar shows whether the API and configured Ollama
model are reachable.
