"""Tests for atomic session acquisition."""

from __future__ import annotations

import pytest

from factcheck.config import AppSettings
from factcheck.db import session_store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", db_path)
    settings = AppSettings(sqlite_path=str(db_path), _env_file=None)
    monkeypatch.setattr("factcheck.config.get_settings", lambda: settings)
    session_store.ensure_dialogue_tables(db_path)
    return db_path


def test_try_acquire_session_succeeds_from_done(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-done",
        raw_input="Done claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )

    assert session_store.try_acquire_session("sess-done", db_path=temp_db) is True

    session = session_store.get_session("sess-done", db_path=temp_db)
    assert session is not None
    assert session["status"] == "running"
    assert session["error"] is None


def test_try_acquire_session_fails_when_running(temp_db) -> None:
    session_store.create_session("sess-running", "Still running.", db_path=temp_db)

    assert session_store.try_acquire_session("sess-running", db_path=temp_db) is False

    session = session_store.get_session("sess-running", db_path=temp_db)
    assert session is not None
    assert session["status"] == "running"


def test_try_acquire_session_fails_when_error(temp_db) -> None:
    session_store.create_session("sess-error", "Failed.", db_path=temp_db)
    session_store.update_session_status(
        "sess-error",
        "error",
        error="Pipeline failed.",
        db_path=temp_db,
    )

    assert session_store.try_acquire_session("sess-error", db_path=temp_db) is False


def test_double_acquire_fails(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-double",
        raw_input="Done claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )

    assert session_store.try_acquire_session("sess-double", db_path=temp_db) is True
    assert session_store.try_acquire_session("sess-double", db_path=temp_db) is False


def test_try_acquire_session_fails_for_missing_session(temp_db) -> None:
    assert session_store.try_acquire_session("missing", db_path=temp_db) is False
