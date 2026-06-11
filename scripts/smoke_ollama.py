#!/usr/bin/env python3
"""Verify that the configured Ollama host is reachable and has the target model."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "backend" / ".env"


def _load_env_file() -> None:
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


async def main() -> int:
    _load_env_file()

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "mistral:7b")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            tags_response = await client.get(f"{base_url}/api/tags")
            tags_response.raise_for_status()
            tags_payload = tags_response.json()
    except Exception as exc:  # pragma: no cover - exercised manually against Ollama
        print(f"Ollama unreachable at {base_url}: {exc}")
        return 1

    models = tags_payload.get("models", [])
    model_names = {item.get("name") for item in models}

    if model not in model_names:
        print(f"Model {model!r} not found at {base_url}. Available models: {sorted(model_names)}")
        return 1

    print(f"Ollama reachable at {base_url}; model {model!r} is available.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
