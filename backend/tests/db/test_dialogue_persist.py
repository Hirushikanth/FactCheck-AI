"""Tests for optimistic dialogue persistence."""

from __future__ import annotations

import sqlite3
import time

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


def test_persist_dialogue_state_enriches_optimistic_user(temp_db) -> None:
    session_store.save_factcheck_session(
        "sess-persist",
        raw_input="Done claim.",
        claim_results=[],
        final_report="Report.",
        db_path=temp_db,
    )
    user_ts = time.time()
    session_store.save_user_message("sess-persist", "Tell me more.", db_path=temp_db)

    enriched_user = {
        "role": "user",
        "content": "Rewritten tell me more.",
        "timestamp": user_ts,
        "intent": "clarification",
        "token_estimate": 4,
    }
    assistant = {
        "role": "assistant",
        "content": "Here is more.",
        "timestamp": user_ts + 1,
        "intent": None,
        "token_estimate": 3,
    }

    session_store.persist_dialogue_state(
        "sess-persist",
        {
            "response": "Here is more.",
            "intent": "clarification",
            "dialogue_history": [enriched_user, assistant],
            "conversation_summary": None,
            "compressed_fc_context": None,
            "fc_context_covers_sequence": None,
            "needs_new_factcheck": False,
            "new_claim_text": None,
            "error": None,
        },
        prior_history_len=1,
        db_path=temp_db,
    )

    with sqlite3.connect(temp_db) as conn:
        rows = conn.execute(
            """
            SELECT role, content, intent
            FROM dialogue_history
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            ("sess-persist",),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0] == ("user", "Rewritten tell me more.", "clarification")
    assert rows[1] == ("assistant", "Here is more.", None)
