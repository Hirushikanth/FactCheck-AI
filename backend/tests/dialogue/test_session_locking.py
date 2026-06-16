"""Tests for dialogue session locking lifecycle."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from factcheck.config import AppSettings
from factcheck.db import session_store
from factcheck.dialogue import service as dialogue_service


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", db_path)
    settings = AppSettings(dev_stream_enabled=False, sqlite_path=str(db_path), _env_file=None)
    monkeypatch.setattr("factcheck.config.get_settings", lambda: settings)
    session_store.ensure_dialogue_tables(db_path)
    return db_path


def _seed_done_session(session_id: str, db_path) -> None:
    session_store.save_factcheck_session(
        session_id,
        raw_input="The Earth is round.",
        claim_results=[],
        final_report="Report.",
        db_path=db_path,
    )
    session_store.update_session_status(session_id, "running", db_path=db_path)


@pytest.mark.asyncio
async def test_run_dialogue_turn_background_sets_done_on_success(temp_db, monkeypatch) -> None:
    _seed_done_session("sess-bg", temp_db)

    async def fake_run_dialogue_with_events(**kwargs):
        return {
            "response": "Answer.",
            "intent": "clarification",
            "dialogue_history": [],
            "conversation_summary": None,
            "compressed_fc_context": None,
            "needs_new_factcheck": False,
            "new_claim_text": None,
            "error": None,
        }

    monkeypatch.setattr(
        dialogue_service,
        "run_dialogue_with_events",
        fake_run_dialogue_with_events,
    )
    monkeypatch.setattr(dialogue_service, "persist_dialogue_state", lambda *args, **kwargs: None)

    await dialogue_service.run_dialogue_turn_background("sess-bg", "Hello?")

    session = session_store.get_session("sess-bg", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"


@pytest.mark.asyncio
async def test_run_dialogue_turn_background_sets_error_on_failure(temp_db, monkeypatch) -> None:
    _seed_done_session("sess-bg-err", temp_db)

    async def fake_run_dialogue_with_events(**kwargs):
        raise RuntimeError("Dialogue failed.")

    monkeypatch.setattr(
        dialogue_service,
        "run_dialogue_with_events",
        fake_run_dialogue_with_events,
    )

    await dialogue_service.run_dialogue_turn_background("sess-bg-err", "Hello?")

    session = session_store.get_session("sess-bg-err", db_path=temp_db)
    assert session is not None
    assert session["status"] == "error"
    assert "Dialogue failed." in (session["error"] or "")


@pytest.mark.asyncio
async def test_trigger_new_factcheck_sets_done_via_save(temp_db, monkeypatch) -> None:
    session_store.create_session("sess-refact", "Original.", db_path=temp_db)

    async def fake_run_factcheck_with_events(**kwargs):
        return {
            "claim_results": [{"claim": "New claim.", "verdict": "SUPPORTED"}],
            "final_report": "New report.",
        }

    monkeypatch.setattr(
        dialogue_service,
        "run_factcheck_with_events",
        fake_run_factcheck_with_events,
    )
    monkeypatch.setattr(dialogue_service, "create_session_queue", lambda session_id: None)

    await dialogue_service._trigger_new_factcheck("sess-refact", "New claim.")

    session = session_store.get_session("sess-refact", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    assert session["raw_input"] == "New claim."


@pytest.mark.asyncio
async def test_trigger_new_factcheck_does_not_call_try_acquire(temp_db, monkeypatch) -> None:
    session_store.create_session("sess-no-acquire", "Original.", db_path=temp_db)

    async def fake_run_factcheck_with_events(**kwargs):
        return {"claim_results": [], "final_report": "Report."}

    monkeypatch.setattr(
        dialogue_service,
        "run_factcheck_with_events",
        fake_run_factcheck_with_events,
    )
    monkeypatch.setattr(dialogue_service, "create_session_queue", lambda session_id: None)

    with patch.object(session_store, "try_acquire_session") as store_acquire:
        await dialogue_service._trigger_new_factcheck("sess-no-acquire", "New claim.")
        store_acquire.assert_not_called()
