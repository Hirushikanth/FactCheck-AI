# Ollama Setup

FactCheck AI uses Ollama to run `gemma4` locally. The backend switches between local and LAN inference by changing only `OLLAMA_BASE_URL`.

The original proposal referenced Qwen 2.5 3B; development moved to Mistral 7B for more reliable structured verifier outputs, and the current default is `gemma4`.

## Mode A: MacBook Local

Use this mode when Ollama runs on the same MacBook as the backend.

```bash
ollama pull gemma4
ollama serve
```

Backend environment:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4
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
ollama pull gemma4
```

On the MacBook backend:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
OLLAMA_MODEL=gemma4
```

For reliability, reserve the Windows PC IP address in the router DHCP settings or configure a static IP.

## Concurrency

Keep `OLLAMA_CONCURRENCY=1` for a MacBook or consumer GPU. The backend may still schedule extractor work with `asyncio.gather`, but every Ollama request is capped at the HTTP layer so only one local model request runs at a time.

Try `OLLAMA_CONCURRENCY=2` only on high-VRAM hardware, such as a 24 GB+ GPU or a remote LAN host with enough memory for multiple model contexts. Higher values can cause VRAM thrashing or out-of-memory failures.

Selection and disambiguation schedule voting work across sentences with `asyncio.gather`; the semaphore still enforces `OLLAMA_CONCURRENCY`. Within each sentence, all `completions` runs finish before majority voting, so `completions=3` with `min_successes=2` issues three Ollama calls per sentence and accepts the output only when at least two normalized responses agree.

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
- `model_loaded: false`: run `ollama pull gemma4` on the active Ollama host.
- Mode B timeout: check Windows Firewall and confirm the MacBook can ping or curl the Windows LAN IP.
- Mode B breaks after reboot: configure a DHCP reservation or static IP.
- First request is slow: increase `OLLAMA_TIMEOUT`; the default is `120` seconds.
