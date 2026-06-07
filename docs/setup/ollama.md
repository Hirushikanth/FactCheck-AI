# Ollama Setup

FactCheck AI uses Ollama to run `mistral:7b` locally. The backend switches between local and LAN inference by changing only `OLLAMA_BASE_URL`.

The project proposal originally referenced Qwen 2.5 3B, but the implementation uses Mistral 7B because it produced more reliable structured verifier outputs during development.

## Mode A: MacBook Local

Use this mode when Ollama runs on the same MacBook as the backend.

```bash
ollama pull mistral:7b
ollama serve
```

Backend environment:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

Expected later end-to-end latency is roughly 15-45 seconds for a multi-claim input on a capable Apple Silicon machine. The first request can be slower because the model may cold start.

## Mode B: Windows PC on LAN

Use this mode when a Windows PC on the same local network hosts Ollama, ideally with a discrete GPU.

On the Windows PC:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0','Machine')
```

Then restart Ollama, allow inbound TCP traffic on port `11434` in Windows Defender Firewall, and pull the model:

```powershell
ollama pull mistral:7b
```

On the MacBook backend:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
OLLAMA_MODEL=mistral:7b
```

For reliability, reserve the Windows PC IP address in the router DHCP settings or configure a static IP.

## Concurrency

Keep `OLLAMA_CONCURRENCY=1` for a MacBook or consumer GPU. The backend may still schedule extractor work with `asyncio.gather`, but every Ollama request is capped at the HTTP layer so only one local model request runs at a time.

Try `OLLAMA_CONCURRENCY=2` only on high-VRAM hardware, such as a 24 GB+ GPU or a remote LAN host with enough memory for multiple model contexts. Higher values can cause VRAM thrashing or out-of-memory failures.

Voting stages run repeated completions sequentially. If a stage uses `completions=3`, expect roughly three serial Ollama calls for that sentence; this is slower but keeps local inference predictable.

## Smoke Test

From the backend virtual environment:

```bash
poetry run python ../scripts/smoke_ollama.py
```

Manual check:

```bash
curl http://localhost:11434/api/tags
curl http://<windows-lan-ip>:11434/api/tags
```

## Troubleshooting

- `Connection refused`: Ollama is not running or the URL is wrong.
- `model_loaded: false`: run `ollama pull mistral:7b` on the active Ollama host.
- Mode B timeout: check Windows Firewall and confirm the MacBook can ping or curl the Windows LAN IP.
- Mode B breaks after reboot: configure a DHCP reservation or static IP.
- First request is slow: increase `OLLAMA_TIMEOUT`; the default is `120` seconds.
