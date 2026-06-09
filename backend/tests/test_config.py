from factcheck.config import AppSettings


def test_settings_defaults_support_local_ollama(monkeypatch) -> None:
    for env_name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TEMPERATURE",
        "OLLAMA_TIMEOUT",
        "OLLAMA_MAX_RETRIES",
        "OLLAMA_NUM_CTX",
        "OLLAMA_CONCURRENCY",
        "SEARCH_MAX_RESULTS",
        "SEARCH_PROVIDER_ORDER",
        "TAVILY_API_KEY",
        "SERPER_API_KEY",
        "DEV_STREAM_ENABLED",
        "DEV_CORS_ORIGINS",
        "DEBUG",
    ):
        monkeypatch.delenv(env_name, raising=False)

    settings = AppSettings(_env_file=None)

    assert str(settings.ollama_base_url) == "http://localhost:11434"
    assert settings.ollama_model == "mistral:7b"
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_timeout == 120
    assert settings.ollama_max_retries == 3
    assert settings.ollama_num_ctx is None
    assert settings.ollama_concurrency == 1
    assert settings.search_max_results == 5
    assert settings.search_provider_order == "duckduckgo,tavily,serper"
    assert settings.tavily_api_key is None
    assert settings.serper_api_key is None
    assert settings.dev_stream_enabled is False
    assert settings.dev_cors_origins == (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    )
    assert settings.debug is False


def test_ollama_concurrency_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_CONCURRENCY", "2")

    settings = AppSettings(_env_file=None)

    assert settings.ollama_concurrency == 2


def test_ollama_num_ctx_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    settings = AppSettings(_env_file=None)

    assert settings.ollama_num_ctx == 8192
