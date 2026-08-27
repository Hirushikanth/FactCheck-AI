"""Environment-driven application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class AppSettings(BaseSettings):
    """Runtime settings loaded from environment variables and backend/.env."""

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = "gemma4"
    ollama_temperature: float = 0.0
    ollama_timeout: int = 120
    # Total attempts per remote Ollama call, including the initial attempt.
    ollama_max_retries: int = Field(default=3, ge=1)
    ollama_num_ctx: int | None = Field(default=None, ge=1)
    # Maximum number of concurrent in-flight requests sent to Ollama.
    # Keep at 1 for consumer GPUs; raise to 2 only on high-VRAM machines.
    ollama_concurrency: int = Field(default=1, ge=1)
    search_max_results: int = 5
    search_provider_order: str = "duckduckgo,tavily,serper"
    search_api_max_retries: int = Field(default=3, ge=1)
    search_api_retry_base_delay: float = Field(default=1.0, ge=0.0)
    search_api_retry_max_delay: float = Field(default=8.0, ge=0.0)
    search_api_timeout_seconds: float = Field(default=30.0, gt=0.0)
    ddg_max_retries: int = Field(default=3, ge=1)
    ddg_retry_base_delay: float = Field(default=1.0, ge=0.0)
    ddg_retry_max_delay: float = Field(default=8.0, ge=0.0)
    ddg_min_request_interval: float = Field(default=1.5, ge=0.0)
    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    # Opt-in guard for examiner demonstrations that must prove fallback works.
    demo_require_fallback: bool = False
    full_page_fetch_mode: Literal["off", "provider", "pinned"] = "provider"
    dev_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    sqlite_path: str = "factcheck_ai.db"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    def validate_search_demo_config(self) -> None:
        """Require a keyed remote fallback in the configured provider order.

        This check is intentionally explicit rather than a model validator so
        DuckDuckGo-only development remains a supported default. Application
        startup calls it only when ``DEMO_REQUIRE_FALLBACK=true``.
        """

        configured_order = {
            item.strip().lower()
            for item in self.search_provider_order.split(",")
            if item.strip()
        }
        keyed_fallbacks = (
            ("tavily", "TAVILY_API_KEY", self.tavily_api_key),
            ("serper", "SERPER_API_KEY", self.serper_api_key),
        )
        if any(
            provider in configured_order and bool(api_key and api_key.strip())
            for provider, _, api_key in keyed_fallbacks
        ):
            return

        missing_keys = [
            env_name
            for provider, env_name, api_key in keyed_fallbacks
            if provider in configured_order and not (api_key and api_key.strip())
        ]
        if missing_keys:
            detail = ", ".join(missing_keys)
        else:
            detail = "TAVILY_API_KEY or SERPER_API_KEY"
        raise ValueError(
            "DEMO_REQUIRE_FALLBACK=true requires at least one configured keyed "
            "fallback included in SEARCH_PROVIDER_ORDER; missing "
            f"{detail}."
        )


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()
