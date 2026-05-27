import httpx
import pytest

from factcheck.config import AppSettings
from factcheck.llm.ollama import check_ollama_health


@pytest.mark.asyncio
async def test_ollama_health_reports_available_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})

    settings = AppSettings()
    transport = httpx.MockTransport(handler)

    health = await check_ollama_health(settings=settings, transport=transport)

    assert health["reachable"] is True
    assert health["model_loaded"] is True
    assert health["base_url"] == "http://localhost:11434"
    assert health["model"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_ollama_health_reports_missing_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2"}]})

    health = await check_ollama_health(settings=AppSettings(), transport=httpx.MockTransport(handler))

    assert health["reachable"] is True
    assert health["model_loaded"] is False
