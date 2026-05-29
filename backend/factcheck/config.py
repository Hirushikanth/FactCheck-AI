"""Environment-driven application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class AppSettings(BaseSettings):
    """Runtime settings loaded from environment variables and backend/.env."""

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.0
    ollama_timeout: int = 120
    ollama_max_retries: int = 3
    search_max_results: int = 5
    search_provider_order: str = "duckduckgo,tavily,serper"
    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    dev_stream_enabled: bool = False
    dev_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()
