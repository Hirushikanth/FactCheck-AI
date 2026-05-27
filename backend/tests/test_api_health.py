from fastapi.testclient import TestClient

from app import main


def test_health_endpoint_shape() -> None:
    async def fake_health() -> dict[str, bool | str]:
        return {
            "status": "ok",
            "ollama_reachable": True,
            "model_loaded": True,
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "qwen2.5:3b",
        }

    original = main.build_health_payload
    main.build_health_payload = fake_health
    client = TestClient(main.app)

    try:
        response = client.get("/api/health")
    finally:
        main.build_health_payload = original

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ollama_reachable": True,
        "model_loaded": True,
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "qwen2.5:3b",
    }


def test_health_payload_reports_unreachable_ollama(monkeypatch) -> None:
    async def fake_check_ollama_health() -> dict[str, bool | str]:
        return {
            "reachable": False,
            "model_loaded": False,
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:3b",
        }

    monkeypatch.setattr(main, "check_ollama_health", fake_check_ollama_health)

    payload = client_payload(main.build_health_payload)

    assert payload["status"] == "error"
    assert payload["ollama_reachable"] is False
    assert payload["model_loaded"] is False


def client_payload(coro):
    import asyncio

    return asyncio.run(coro())
