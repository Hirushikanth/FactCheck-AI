"""Tests for fact-check runs storage."""

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


def test_create_session_creates_initial_run(temp_db) -> None:
    run_id = session_store.create_session("sess-1", "Original claim.", db_path=temp_db)

    runs = session_store.get_factcheck_runs("sess-1", db_path=temp_db)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["sequence"] == 1
    assert runs[0]["status"] == "running"
    assert runs[0]["triggered_by"] == "initial"

    session = session_store.get_session("sess-1", db_path=temp_db)
    assert session is not None
    assert session["active_run_id"] == run_id
    assert session["status"] == "running"


def test_dialogue_refactcheck_appends_run_preserves_original(temp_db) -> None:
    session_store.create_session("sess-2", "Original claim.", db_path=temp_db)
    run1_id = session_store.get_initial_run_id("sess-2", db_path=temp_db)
    assert run1_id is not None
    session_store.complete_factcheck_run(
        run1_id,
        claim_results=[{"claim": "Original claim.", "verdict": "SUPPORTED"}],
        final_report="Original report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-2", run1_id, db_path=temp_db)
    session_store.update_session_status("sess-2", "done", db_path=temp_db)

    run2_id = session_store.create_factcheck_run(
        "sess-2", "New claim.", "dialogue", db_path=temp_db
    )
    session_store.complete_factcheck_run(
        run2_id,
        claim_results=[{"claim": "New claim.", "verdict": "REFUTED"}],
        final_report="New report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-2", run2_id, db_path=temp_db)

    runs = session_store.get_factcheck_runs("sess-2", db_path=temp_db)
    assert len(runs) == 2
    assert runs[0]["raw_input"] == "Original claim."
    assert runs[0]["claim_results"][0]["verdict"] == "SUPPORTED"
    assert runs[1]["raw_input"] == "New claim."
    assert runs[1]["claim_results"][0]["verdict"] == "REFUTED"

    session = session_store.get_session("sess-2", db_path=temp_db)
    assert session is not None
    assert session["raw_input"] == "New claim."
    assert session["active_run_id"] == run2_id
    assert len(session["runs"]) == 2


def test_stale_fc_context_omitted_from_dialogue_load(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-stale",
        raw_input="Original.",
        claim_results=[{"claim": "A", "verdict": "SUPPORTED"}],
        final_report="Report.",
        db_path=temp_db,
    )
    run2_id = session_store.create_factcheck_run(
        "sess-stale", "New.", "dialogue", db_path=temp_db
    )
    session_store.complete_factcheck_run(
        run2_id,
        claim_results=[{"claim": "B", "verdict": "REFUTED"}],
        final_report="New report.",
        db_path=temp_db,
    )
    session_store.set_active_run("sess-stale", run2_id, db_path=temp_db)

    with session_store._get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO dialogue_fc_context
              (session_id, compressed_context, created_at, covers_through_sequence)
            VALUES ('sess-stale', 'stale cache', ?, 1)
            """,
            (__import__("time").time(),),
        )

    loaded = session_store.load_session_for_dialogue("sess-stale", db_path=temp_db)
    assert loaded["compressed_fc_context"] is None
    assert loaded["latest_run_sequence"] == 2
    assert len(loaded["fact_check_runs"]) == 2


def test_invalidate_fc_context(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-inv",
        raw_input="Claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )
    with session_store._get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO dialogue_fc_context
              (session_id, compressed_context, created_at, covers_through_sequence)
            VALUES ('sess-inv', 'cached', ?, 1)
            """,
            (__import__("time").time(),),
        )

    session_store.invalidate_fc_context("sess-inv", db_path=temp_db)
    loaded = session_store.load_session_for_dialogue("sess-inv", db_path=temp_db)
    assert loaded["compressed_fc_context"] is None


def test_delete_session_cascades_runs(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-del",
        raw_input="Claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )
    session_store.create_factcheck_run("sess-del", "Second.", "dialogue", db_path=temp_db)

    assert session_store.delete_session("sess-del", db_path=temp_db) is True
    assert session_store.get_session("sess-del", db_path=temp_db) is None
    assert session_store.get_factcheck_runs("sess-del", db_path=temp_db) == []


def test_list_sessions_uses_first_run_input(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-list",
        raw_input="First claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )
    run2_id = session_store.create_factcheck_run(
        "sess-list", "Second claim.", "dialogue", db_path=temp_db
    )
    session_store.complete_factcheck_run(
        run2_id, claim_results=[], final_report="R2", db_path=temp_db
    )
    session_store.set_active_run("sess-list", run2_id, db_path=temp_db)

    summaries = session_store.list_sessions(db_path=temp_db)
    match = next(s for s in summaries if s["session_id"] == "sess-list")
    assert match["raw_input"] == "First claim."
