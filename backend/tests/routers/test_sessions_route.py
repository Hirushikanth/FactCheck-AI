"""Tests for session API routes."""

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


def test_create_session_returns_202_and_running_status(client, temp_db, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.routers.sessions._run_and_persist",
        AsyncMock(),
    )

    response = client.post(
        "/api/sessions",
        json={"input": "The Earth is round."},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert body["session_id"]

    session = session_store.get_session(body["session_id"])
    assert session is not None
    assert session["status"] == "running"


def test_get_session_returns_404_for_missing(client, temp_db) -> None:
    response = client.get("/api/sessions/missing-id")
    assert response.status_code == 404


def test_post_message_returns_409_when_not_done(client, temp_db) -> None:
    session_store.create_session("sess-running", "Still running.", db_path=temp_db)

    response = client.post(
        "/api/sessions/sess-running/messages",
        json={"message": "Follow-up?"},
    )

    assert response.status_code == 409


def test_post_message_accepts_done_session(client, temp_db, monkeypatch) -> None:
    session_store.save_factcheck_session(
        "sess-done",
        raw_input="Done claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )

    monkeypatch.setattr(
        "app.routers.sessions.run_dialogue_turn_background",
        AsyncMock(),
    )

    response = client.post(
        "/api/sessions/sess-done/messages",
        json={"message": "Tell me more."},
    )

    assert response.status_code == 202
    assert response.json()["message_id"]


def test_list_and_delete_sessions(client, temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-list",
        raw_input="List me.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )
    assert any(
        item["session_id"] == "sess-list"
        for item in session_store.list_sessions(db_path=temp_db)
    )

    list_response = client.get("/api/sessions")
    assert list_response.status_code == 200
    summaries = list_response.json()
    assert any(item["session_id"] == "sess-list" for item in summaries)

    delete_response = client.delete("/api/sessions/sess-list")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert session_store.get_session("sess-list") is None
