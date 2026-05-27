# Ollama Setup

FactCheck AI uses Ollama to run `qwen2.5:3b` locally. The backend switches between local and LAN inference by changing only `OLLAMA_BASE_URL`.

## Mode A: MacBook Local

Use this mode when Ollama runs on the same MacBook as the backend.

```bash
ollama pull qwen2.5:3b
ollama serve
```

Backend environment:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
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
ollama pull qwen2.5:3b
```

On the MacBook backend:

```bash
OLLAMA_BASE_URL=http://<windows-lan-ip>:11434
OLLAMA_MODEL=qwen2.5:3b
```

For reliability, reserve the Windows PC IP address in the router DHCP settings or configure a static IP.

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
- `model_loaded: false`: run `ollama pull qwen2.5:3b` on the active Ollama host.
- Mode B timeout: check Windows Firewall and confirm the MacBook can ping or curl the Windows LAN IP.
- Mode B breaks after reboot: configure a DHCP reservation or static IP.
- First request is slow: increase `OLLAMA_TIMEOUT`; the default is `120` seconds.
