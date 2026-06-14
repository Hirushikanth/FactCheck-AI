"""SQLite session store for fact-check and dialogue persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from factcheck.config import BACKEND_DIR
from factcheck.dialogue.schemas import ConversationSummary, DialogueOutput, DialogueTurn

DEFAULT_DB_PATH = BACKEND_DIR / "factcheck_ai.db"


def _get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_dialogue_tables(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create dialogue and session tables if they do not exist."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_check_sessions (
                session_id       TEXT PRIMARY KEY,
                raw_input        TEXT NOT NULL,
                claim_results_json TEXT NOT NULL DEFAULT '[]',
                final_report     TEXT,
                created_at       REAL NOT NULL,
                updated_at       REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialogue_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                intent          TEXT,
                token_estimate  INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES fact_check_sessions(session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialogue_summaries (
                session_id        TEXT PRIMARY KEY,
                summary_text      TEXT NOT NULL,
                turns_compressed  INTEGER NOT NULL,
                last_updated      REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES fact_check_sessions(session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialogue_fc_context (
                session_id          TEXT PRIMARY KEY,
                compressed_context  TEXT NOT NULL,
                created_at          REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES fact_check_sessions(session_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dialogue_history_session "
            "ON dialogue_history(session_id)"
        )


def session_exists(session_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    ensure_dialogue_tables(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM fact_check_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None


def save_factcheck_session(
    session_id: str,
    *,
    raw_input: str,
    claim_results: list[dict[str, Any]],
    final_report: str | None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Create or update a completed fact-check session snapshot."""
    ensure_dialogue_tables(db_path)
    now = time.time()
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO fact_check_sessions
              (session_id, raw_input, claim_results_json, final_report, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              raw_input = excluded.raw_input,
              claim_results_json = excluded.claim_results_json,
              final_report = excluded.final_report,
              updated_at = excluded.updated_at
            """,
            (
                session_id,
                raw_input,
                json.dumps(claim_results),
                final_report,
                now,
                now,
            ),
        )


def load_session_for_dialogue(
    session_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Load fact-check snapshot and dialogue state for a session."""
    ensure_dialogue_tables(db_path)
    with _get_connection(db_path) as conn:
        fc_row = conn.execute(
            "SELECT * FROM fact_check_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if fc_row is None:
            raise KeyError(session_id)

        history_rows = conn.execute(
            "SELECT * FROM dialogue_history WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()

        summary_row = conn.execute(
            "SELECT * FROM dialogue_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        fc_context_row = conn.execute(
            "SELECT compressed_context FROM dialogue_fc_context WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    dialogue_history: list[DialogueTurn] = [
        DialogueTurn(
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            intent=row["intent"],
            token_estimate=row["token_estimate"] or 0,
        )
        for row in history_rows
    ]

    conversation_summary: ConversationSummary | None = None
    if summary_row is not None:
        conversation_summary = ConversationSummary(
            text=summary_row["summary_text"],
            turns_compressed=summary_row["turns_compressed"],
            last_updated=summary_row["last_updated"],
        )

    return {
        "raw_input": fc_row["raw_input"],
        "claim_results": json.loads(fc_row["claim_results_json"]),
        "final_report": fc_row["final_report"],
        "dialogue_history": dialogue_history,
        "conversation_summary": conversation_summary,
        "compressed_fc_context": (
            fc_context_row["compressed_context"] if fc_context_row is not None else None
        ),
    }


def persist_dialogue_state(
    session_id: str,
    result: DialogueOutput,
    *,
    prior_history_len: int,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Persist dialogue deltas after one turn."""
    ensure_dialogue_tables(db_path)
    new_turns = result["dialogue_history"][prior_history_len:]

    with _get_connection(db_path) as conn:
        for turn in new_turns:
            conn.execute(
                """
                INSERT INTO dialogue_history
                  (session_id, role, content, timestamp, intent, token_estimate)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn["role"],
                    turn["content"],
                    turn["timestamp"],
                    turn.get("intent"),
                    turn.get("token_estimate", 0),
                ),
            )

        summary = result.get("conversation_summary")
        if summary is not None:
            conn.execute(
                """
                INSERT INTO dialogue_summaries
                  (session_id, summary_text, turns_compressed, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  summary_text = excluded.summary_text,
                  turns_compressed = excluded.turns_compressed,
                  last_updated = excluded.last_updated
                """,
                (
                    session_id,
                    summary["text"],
                    summary["turns_compressed"],
                    summary["last_updated"],
                ),
            )

        compressed = result.get("compressed_fc_context")
        if compressed:
            conn.execute(
                """
                INSERT INTO dialogue_fc_context
                  (session_id, compressed_context, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, compressed, time.time()),
            )
