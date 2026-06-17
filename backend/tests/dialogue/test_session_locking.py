"""Tests for dialogue session locking lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from factcheck.config import AppSettings
from factcheck.db import session_store
from factcheck.dialogue import service as dialogue_service


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", db_path)
    settings = AppSettings(sqlite_path=str(db_path), _env_file=None)
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
async def test_run_dialogue_turn_background_persists_without_duplicate_users(
    temp_db, monkeypatch
) -> None:
    _seed_done_session("sess-opt", temp_db)
    session_store.save_user_message("sess-opt", "Hello?", db_path=temp_db)

    async def fake_run_dialogue_with_events(**kwargs):
        history = list(kwargs["dialogue_history"])
        user_ts = history[-1]["timestamp"]
        enriched_user = {
            "role": "user",
            "content": "Hello?",
            "timestamp": user_ts,
            "intent": "clarification",
            "token_estimate": 2,
        }
        assistant = {
            "role": "assistant",
            "content": "Answer.",
            "timestamp": user_ts + 1,
            "intent": None,
            "token_estimate": 1,
        }
        return {
            "response": "Answer.",
            "intent": "clarification",
            "dialogue_history": history[:-1] + [enriched_user, assistant],
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

    await dialogue_service.run_dialogue_turn_background("sess-opt", "Hello?")

    session = session_store.get_session("sess-opt", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    assert len(session["messages"]) == 2
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][0]["content"] == "Hello?"
    assert session["messages"][1]["role"] == "assistant"
    assert session["messages"][1]["content"] == "Answer."


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
async def test_trigger_new_factcheck_appends_run_preserves_original(temp_db, monkeypatch) -> None:
    run1_id = session_store.create_session("sess-refact", "Original.", db_path=temp_db)
    session_store.complete_factcheck_run(
        run1_id,
        claim_results=[{"claim": "Original.", "verdict": "SUPPORTED"}],
        final_report="Original report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-refact", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-refact", "done", db_path=temp_db)

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
    monkeypatch.setattr(dialogue_service, "create_session_hub", lambda session_id, run_id=None: None)

    await dialogue_service._trigger_new_factcheck("sess-refact", "New claim.")

    session = session_store.get_session("sess-refact", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    assert session["raw_input"] == "New claim."
    runs = session_store.get_factcheck_runs("sess-refact", db_path=temp_db)
    assert len(runs) == 2
    assert runs[0]["raw_input"] == "Original."
    assert runs[1]["raw_input"] == "New claim."


@pytest.mark.asyncio
async def test_trigger_new_factcheck_calls_try_acquire(temp_db, monkeypatch) -> None:
    run1_id = session_store.create_session("sess-acquire", "Original.", db_path=temp_db)
    session_store.complete_factcheck_run(
        run1_id, claim_results=[], final_report="Report.", db_path=temp_db
    )
    session_store.set_active_run("sess-acquire", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-acquire", "done", db_path=temp_db)

    async def fake_run_factcheck_with_events(**kwargs):
        return {"claim_results": [], "final_report": "Report."}

    monkeypatch.setattr(
        dialogue_service,
        "run_factcheck_with_events",
        fake_run_factcheck_with_events,
    )
    monkeypatch.setattr(dialogue_service, "create_session_hub", lambda session_id, run_id=None: None)

    with patch.object(dialogue_service, "try_acquire_session", return_value=True) as store_acquire:
        await dialogue_service._trigger_new_factcheck("sess-acquire", "New claim.")
        store_acquire.assert_called_once_with("sess-acquire")


@pytest.mark.asyncio
async def test_trigger_new_factcheck_skips_acquire_when_lock_held(
    temp_db, monkeypatch
) -> None:
    run1_id = session_store.create_session("sess-lock-held", "Original.", db_path=temp_db)
    session_store.complete_factcheck_run(
        run1_id,
        claim_results=[{"claim": "Original.", "verdict": "SUPPORTED"}],
        final_report="Original report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-lock-held", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-lock-held", "running", db_path=temp_db)

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
    monkeypatch.setattr(dialogue_service, "create_session_hub", lambda session_id, run_id=None: None)

    with patch.object(dialogue_service, "try_acquire_session") as store_acquire:
        await dialogue_service._trigger_new_factcheck(
            "sess-lock-held", "New claim.", lock_held=True
        )
        store_acquire.assert_not_called()

    session = session_store.get_session("sess-lock-held", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    runs = session_store.get_factcheck_runs("sess-lock-held", db_path=temp_db)
    assert len(runs) == 2
    assert runs[1]["raw_input"] == "New claim."


@pytest.mark.asyncio
async def test_run_dialogue_turn_triggers_factcheck_with_lock_held(
    temp_db, monkeypatch
) -> None:
    run1_id = session_store.create_session("sess-sync-fc", "Original.", db_path=temp_db)
    session_store.complete_factcheck_run(
        run1_id,
        claim_results=[{"claim": "Original.", "verdict": "SUPPORTED"}],
        final_report="Original report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-sync-fc", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-sync-fc", "done", db_path=temp_db)

    async def fake_run_dialogue(**kwargs):
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

    async def fake_run_factcheck_with_events(**kwargs):
        return {
            "claim_results": [{"claim": "The moon is cheese.", "verdict": "REFUTED"}],
            "final_report": "Moon report.",
        }

    monkeypatch.setattr(dialogue_service, "run_dialogue", fake_run_dialogue)
    monkeypatch.setattr(
        dialogue_service,
        "run_factcheck_with_events",
        fake_run_factcheck_with_events,
    )
    monkeypatch.setattr(dialogue_service, "create_session_hub", lambda session_id, run_id=None: None)
    monkeypatch.setattr(dialogue_service, "persist_dialogue_state", lambda *args, **kwargs: None)

    await dialogue_service.run_dialogue_turn("sess-sync-fc", "Check: The moon is cheese.")
    await asyncio.sleep(0)

    session = session_store.get_session("sess-sync-fc", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    runs = session_store.get_factcheck_runs("sess-sync-fc", db_path=temp_db)
    assert len(runs) == 2
    assert runs[1]["raw_input"] == "The moon is cheese."


@pytest.mark.asyncio
async def test_run_dialogue_turn_background_triggers_factcheck_when_running(
    temp_db, monkeypatch
) -> None:
    run1_id = session_store.create_session("sess-bg-fc", "Original.", db_path=temp_db)
    session_store.complete_factcheck_run(
        run1_id,
        claim_results=[{"claim": "Original.", "verdict": "SUPPORTED"}],
        final_report="Original report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-bg-fc", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-bg-fc", "running", db_path=temp_db)

    async def fake_run_dialogue_with_events(**kwargs):
        return {
            "response": "Queued.",
            "intent": "new_claim",
            "dialogue_history": [],
            "conversation_summary": None,
            "compressed_fc_context": None,
            "needs_new_factcheck": True,
            "new_claim_text": "New claim.",
            "error": None,
        }

    async def fake_run_factcheck_with_events(**kwargs):
        return {
            "claim_results": [{"claim": "New claim.", "verdict": "SUPPORTED"}],
            "final_report": "New report.",
        }

    monkeypatch.setattr(
        dialogue_service,
        "run_dialogue_with_events",
        fake_run_dialogue_with_events,
    )
    monkeypatch.setattr(
        dialogue_service,
        "run_factcheck_with_events",
        fake_run_factcheck_with_events,
    )
    monkeypatch.setattr(dialogue_service, "create_session_hub", lambda session_id, run_id=None: None)
    monkeypatch.setattr(dialogue_service, "persist_dialogue_state", lambda *args, **kwargs: None)

    with patch.object(dialogue_service, "try_acquire_session") as store_acquire:
        await dialogue_service.run_dialogue_turn_background("sess-bg-fc", "New claim.")
        store_acquire.assert_not_called()

    session = session_store.get_session("sess-bg-fc", db_path=temp_db)
    assert session is not None
    assert session["status"] == "done"
    runs = session_store.get_factcheck_runs("sess-bg-fc", db_path=temp_db)
    assert len(runs) == 2
    assert runs[1]["raw_input"] == "New claim."
