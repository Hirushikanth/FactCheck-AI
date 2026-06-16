"""FastAPI entry point for the FactCheck AI backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import dialogue, sessions
from factcheck.config import AppSettings, get_settings
from factcheck.db.session_store import ensure_dialogue_tables
from factcheck.llm.ollama import check_ollama_health


def _parse_cors_origins(origins: str) -> list[str]:
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    settings = getattr(app.state, "settings", None) or get_settings()
    ensure_dialogue_tables(settings.sqlite_path)
    yield


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create the FastAPI application with optional dev-only routes."""

    resolved_settings = settings or get_settings()
    app = FastAPI(title="FactCheck AI", version="0.6.0", lifespan=lifespan)
    app.state.settings = resolved_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(resolved_settings.dev_cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(sessions.router)
    app.include_router(dialogue.router)

    @app.get("/api/health")
    async def health() -> dict[str, bool | str]:
        """Return backend and Ollama health information."""

        return await build_health_payload()

    return app


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


app = create_app()
