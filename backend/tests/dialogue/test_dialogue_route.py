"""Tests for dialogue API route and session persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from factcheck.config import AppSettings
from factcheck.db import session_store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", db_path)
    settings = AppSettings(dev_stream_enabled=False, sqlite_path=str(db_path), _env_file=None)
    monkeypatch.setattr("factcheck.config.get_settings", lambda: settings)
    session_store.ensure_dialogue_tables(db_path)
    return db_path


@pytest.fixture
def client(temp_db):
    settings = AppSettings(
        dev_stream_enabled=False,
        sqlite_path=str(temp_db),
        _env_file=None,
    )
    return TestClient(create_app(settings=settings))


def _seed_session(session_id: str = "sess-api", db_path=None) -> None:
    session_store.save_factcheck_session(
        session_id,
        raw_input="The Earth is round.",
        claim_results=[
            {
                "claim": "The Earth is round.",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence": ["Round earth evidence."],
                "sources": ["https://example.com/earth"],
                "reasoning": "Supported.",
                "search_queries": ["earth shape"],
            }
        ],
        final_report="# Fact-Check Report\n\nSupported.",
        db_path=db_path,
    )


def test_dialogue_route_returns_404_for_missing_session(client, temp_db) -> None:
    response = client.post(
        "/api/dialogue/missing-session",
        json={"message": "Why was Claim 1 rated SUPPORTED?"},
    )
    assert response.status_code == 404


def test_dialogue_route_returns_409_when_running(client, temp_db) -> None:
    session_store.create_session("sess-running", "Still running.", db_path=temp_db)

    response = client.post(
        "/api/dialogue/sess-running",
        json={"message": "Follow-up?"},
    )

    assert response.status_code == 409


def test_dialogue_route_returns_intent_and_persists_history(client, temp_db, monkeypatch) -> None:
    _seed_session(db_path=temp_db)

    async def fake_run_dialogue_turn(**kwargs):
        message = kwargs["user_message"]
        return {
            "response": "Claim 1 was SUPPORTED.",
            "intent": "clarification",
            "dialogue_history": [
                {
                    "role": "user",
                    "content": message,
                    "timestamp": 1.0,
                    "intent": "clarification",
                    "token_estimate": 5,
                },
                {
                    "role": "assistant",
                    "content": "Claim 1 was SUPPORTED.",
                    "timestamp": 2.0,
                    "intent": None,
                    "token_estimate": 6,
                },
            ],
            "conversation_summary": None,
            "compressed_fc_context": "=== cached ===",
            "needs_new_factcheck": False,
            "new_claim_text": None,
            "error": None,
        }

    monkeypatch.setattr("factcheck.dialogue.service.run_dialogue", fake_run_dialogue_turn)

    response = client.post(
        "/api/dialogue/sess-api",
        json={"message": "What was the verdict?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "clarification"
    assert body["response"] == "Claim 1 was SUPPORTED."

    session = session_store.load_session_for_dialogue("sess-api")
    assert len(session["dialogue_history"]) == 2
    assert session["compressed_fc_context"] == "=== cached ==="

    stored = session_store.get_session("sess-api", db_path=temp_db)
    assert stored is not None
    assert stored["status"] == "done"


def test_dialogue_route_triggers_pipeline_for_new_claim(client, temp_db, monkeypatch) -> None:
    _seed_session("sess-new-claim", db_path=temp_db)

    async def fake_run_dialogue_turn(**kwargs):
        message = kwargs["user_message"]
        return {
            "response": "Queued.",
            "intent": "new_claim",
            "dialogue_history": [],
            "conversation_summary": None,
            "compressed_fc_context": None,
            "needs_new_factcheck": True,
            "new_claim_text": "The moon is cheese.",
            "error": None,
        }

    trigger = AsyncMock()
    monkeypatch.setattr("factcheck.dialogue.service.run_dialogue", fake_run_dialogue_turn)
    monkeypatch.setattr("factcheck.dialogue.service._trigger_new_factcheck", trigger)

    response = client.post(
        "/api/dialogue/sess-new-claim",
        json={"message": "Check this: The moon is cheese."},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "new_claim"
    assert response.json()["needs_new_factcheck"] is True
