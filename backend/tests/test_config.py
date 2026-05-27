from factcheck.config import AppSettings


def test_settings_defaults_support_local_ollama() -> None:
    settings = AppSettings()

    assert str(settings.ollama_base_url) == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:3b"
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_timeout == 120
    assert settings.ollama_max_retries == 3
    assert settings.debug is False
