"""FastAPI entry point for the Phase 1 backend scaffold."""

from __future__ import annotations

from fastapi import FastAPI

from factcheck.llm.ollama import check_ollama_health


app = FastAPI(title="FactCheck AI", version="0.1.0")


@app.get("/api/health")
async def health() -> dict[str, bool | str]:
    """Return backend and Ollama health information."""

    return await build_health_payload()


async def build_health_payload() -> dict[str, bool | str]:
    """Build the health response payload from the configured Ollama service."""

    ollama = await check_ollama_health()
    return {
        "status": "ok" if ollama["reachable"] else "error",
        "ollama_reachable": bool(ollama["reachable"]),
        "model_loaded": bool(ollama["model_loaded"]),
        "ollama_base_url": str(ollama["base_url"]),
        "ollama_model": str(ollama["model"]),
    }
