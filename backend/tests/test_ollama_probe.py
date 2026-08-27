from __future__ import annotations

import httpx

from factcheck.config import AppSettings
from factcheck.llm.ollama import probe_ollama_capabilities


class ChunkedNDJSONStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def _settings() -> AppSettings:
    return AppSettings(ollama_model="gemma4", _env_file=None)


async def test_probe_reports_supported_and_keeps_raw_thinking_opt_in() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4"}]})
        assert request.url.path == "/api/chat"
        payload = request.read()
        assert b'"think":true' in payload
        return httpx.Response(
            200,
            stream=ChunkedNDJSONStream(
                [
                    b'{"message":{"thinking":"Compare ',
                    b'the sources.","content":""}}\n{"message":{"content":"The answer"},',
                    b'"done":true,"eval_count":4}\n',
                ]
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://localhost:11434", transport=transport) as client:
        result = await probe_ollama_capabilities(_settings(), client)

    assert result["status"] == "supported"
    assert result["reachable"] is True
    assert result["model_installed"] is True
    assert result["thinking_chunks"] == 1
    assert result["content_chunks"] == 1
    assert "thinking" not in result
    assert result["eval_count"] == 4


async def test_probe_can_return_raw_thinking_when_explicitly_requested() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4"}]})
        return httpx.Response(
            200,
            content=b'{"message":{"thinking":"A short trace","content":"done"},"done":true}\n',
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    ) as client:
        result = await probe_ollama_capabilities(_settings(), client, show_raw_thinking=True)

    assert result["status"] == "supported"
    assert result["thinking"] == "A short trace"


async def test_probe_reports_unsupported_for_content_only_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4"}]})
        return httpx.Response(
            200,
            content=b'{"message":{"content":"Only answer"},"done":true}\n',
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    ) as client:
        result = await probe_ollama_capabilities(_settings(), client)

    assert result["status"] == "unsupported"
    assert result["thinking_chunks"] == 0


async def test_probe_reports_probe_failed_for_missing_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2"}]})

    async with httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    ) as client:
        result = await probe_ollama_capabilities(_settings(), client)

    assert result["status"] == "probe_failed"
    assert result["reachable"] is True
    assert result["model_installed"] is False


async def test_probe_reports_probe_failed_for_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama unavailable", request=request)

    async with httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    ) as client:
        result = await probe_ollama_capabilities(_settings(), client)

    assert result["status"] == "probe_failed"
    assert result["reachable"] is False
