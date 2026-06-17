"""Tests for session API routes."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from factcheck.config import AppSettings
from factcheck.db import session_store
from factcheck.graph import event_bus


@pytest.fixture(autouse=True)
def clear_hubs():
    event_bus._hubs.clear()
    yield
    event_bus._hubs.clear()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", db_path)
    settings = AppSettings(sqlite_path=str(db_path), _env_file=None)
    monkeypatch.setattr("factcheck.config.get_settings", lambda: settings)
    session_store.ensure_dialogue_tables(db_path)
    return db_path


@pytest.fixture
def client(temp_db):
    settings = AppSettings(
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
    body = response.json()
    assert body["message_id"]

    session = session_store.get_session("sess-done", db_path=temp_db)
    assert session is not None
    assert session["status"] == "running"
    assert len(session["messages"]) == 1
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][0]["content"] == "Tell me more."

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT id FROM dialogue_history WHERE session_id = ? AND role = 'user'",
            ("sess-done",),
        ).fetchone()
    assert row is not None
    assert body["message_id"] == str(row[0])


def test_post_message_atomic_reject_when_running(client, temp_db) -> None:
    session_store.create_session("sess-running", "Still running.", db_path=temp_db)

    response = client.post(
        "/api/sessions/sess-running/messages",
        json={"message": "Follow-up?"},
    )

    assert response.status_code == 409


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


def test_stream_returns_404_for_missing_session(client, temp_db) -> None:
    response = client.get("/api/sessions/missing-id/stream")
    assert response.status_code == 404


def test_stream_returns_409_when_no_hub_and_session_done(client, temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-done-stream",
        raw_input="Done claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )

    response = client.get("/api/sessions/sess-done-stream/stream")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "stream_missed"
    assert detail["session_status"] == "done"


def test_stream_returns_200_with_stream_open_after_create(
    client, temp_db, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.routers.sessions._run_and_persist",
        AsyncMock(),
    )

    async def fake_stream(session_id: str):
        yield f'event: stream_open\ndata: {{"session_id":"{session_id}"}}\n\n'

    monkeypatch.setattr("app.routers.sessions.stream_events", fake_stream)

    create_response = client.post(
        "/api/sessions",
        json={"input": "The Earth is round."},
    )
    session_id = create_response.json()["session_id"]

    response = client.get(f"/api/sessions/{session_id}/stream")

    assert response.status_code == 200
    assert "event: stream_open" in response.text
    assert session_id in response.text
    assert event_bus.get_hub(session_id) is not None

