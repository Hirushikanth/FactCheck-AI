"""SQLite session store for fact-check and dialogue persistence."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from factcheck.config import BACKEND_DIR
from factcheck.dialogue.schemas import ConversationSummary, DialogueOutput, DialogueTurn

DEFAULT_DB_PATH = BACKEND_DIR / "factcheck_ai.db"


def get_sqlite_path() -> Path:
    """Return the configured SQLite database path."""
    from factcheck.config import get_settings

    settings = get_settings()
    configured = Path(settings.sqlite_path)
    if configured.is_absolute():
        return configured
    return BACKEND_DIR / configured


def _resolve_db_path(db_path: Path | str | None) -> Path | str:
    return db_path if db_path is not None else get_sqlite_path()


def _get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_resolve_db_path(db_path)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_session_columns(conn: sqlite3.Connection) -> None:
    """Add Phase 6 status/error and runs-model columns to existing databases."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(fact_check_sessions)").fetchall()
    }
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE fact_check_sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'done'"
        )
    if "error" not in columns:
        conn.execute("ALTER TABLE fact_check_sessions ADD COLUMN error TEXT")
    if "active_run_id" not in columns:
        conn.execute("ALTER TABLE fact_check_sessions ADD COLUMN active_run_id TEXT")


def _migrate_fc_context_columns(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(dialogue_fc_context)").fetchall()
    }
    if "covers_through_sequence" not in columns:
        conn.execute(
            "ALTER TABLE dialogue_fc_context ADD COLUMN covers_through_sequence INTEGER"
        )


def _migrate_legacy_sessions_to_runs(conn: sqlite3.Connection) -> None:
    """One-time migration: legacy session rows -> fact_check_runs run #1."""
    rows = conn.execute(
        """
        SELECT s.session_id, s.raw_input, s.claim_results_json, s.final_report,
               s.status, s.error, s.created_at, s.updated_at
        FROM fact_check_sessions s
        LEFT JOIN fact_check_runs r ON r.session_id = s.session_id
        WHERE r.run_id IS NULL
        """
    ).fetchall()

    for row in rows:
        run_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO fact_check_runs
              (run_id, session_id, sequence, raw_input, claim_results_json,
               final_report, status, error, triggered_by, created_at, updated_at)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?, 'initial', ?, ?)
            """,
            (
                run_id,
                row["session_id"],
                row["raw_input"],
                row["claim_results_json"],
                row["final_report"],
                row["status"],
                row["error"],
                row["created_at"],
                row["updated_at"],
            ),
        )
        conn.execute(
            """
            UPDATE fact_check_sessions
            SET active_run_id = ?
            WHERE session_id = ?
            """,
            (run_id, row["session_id"]),
        )

    if rows:
        conn.execute(
            """
            UPDATE dialogue_fc_context
            SET covers_through_sequence = 1
            WHERE covers_through_sequence IS NULL
            """
        )


def _parse_run_row(row: sqlite3.Row) -> dict[str, Any]:
    run = dict(row)
    run["claim_results"] = json.loads(run.pop("claim_results_json"))
    return run


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "sequence": run["sequence"],
        "raw_input": run["raw_input"],
        "status": run["status"],
        "triggered_by": run["triggered_by"],
        "created_at": run["created_at"],
    }


def _sync_legacy_session_from_run(
    conn: sqlite3.Connection,
    session_id: str,
    run: dict[str, Any],
) -> None:
    """Keep legacy payload columns in sync with the active run."""
    now = time.time()
    conn.execute(
        """
        UPDATE fact_check_sessions
        SET raw_input = ?,
            claim_results_json = ?,
            final_report = ?,
            updated_at = ?
        WHERE session_id = ?
        """,
        (
            run["raw_input"],
            json.dumps(run["claim_results"]),
            run.get("final_report"),
            now,
            session_id,
        ),
    )


def ensure_dialogue_tables(db_path: Path | str | None = None) -> None:
    """Create dialogue and session tables if they do not exist."""
    resolved = _resolve_db_path(db_path)
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_check_sessions (
                session_id       TEXT PRIMARY KEY,
                raw_input        TEXT NOT NULL,
                claim_results_json TEXT NOT NULL DEFAULT '[]',
                final_report     TEXT,
                status           TEXT NOT NULL DEFAULT 'running',
                error            TEXT,
                active_run_id    TEXT,
                created_at       REAL NOT NULL,
                updated_at       REAL NOT NULL
            )
            """
        )
        _migrate_session_columns(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_check_runs (
                run_id             TEXT PRIMARY KEY,
                session_id         TEXT NOT NULL,
                sequence           INTEGER NOT NULL,
                raw_input          TEXT NOT NULL,
                claim_results_json TEXT NOT NULL DEFAULT '[]',
                final_report       TEXT,
                status             TEXT NOT NULL,
                error              TEXT,
                triggered_by       TEXT NOT NULL,
                created_at         REAL NOT NULL,
                updated_at         REAL NOT NULL,
                UNIQUE (session_id, sequence),
                FOREIGN KEY (session_id) REFERENCES fact_check_sessions(session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fact_check_runs_session
            ON fact_check_runs(session_id)
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
                session_id               TEXT PRIMARY KEY,
                compressed_context       TEXT NOT NULL,
                created_at               REAL NOT NULL,
                covers_through_sequence  INTEGER,
                FOREIGN KEY (session_id) REFERENCES fact_check_sessions(session_id)
            )
            """
        )
        _migrate_fc_context_columns(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dialogue_history_session "
            "ON dialogue_history(session_id)"
        )
        _migrate_legacy_sessions_to_runs(conn)


def create_session(
    session_id: str,
    raw_input: str,
    db_path: Path | str | None = None,
) -> str:
    """Insert a new session with status='running' and initial run #1."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    run_id = str(uuid.uuid4())
    now = time.time()
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            INSERT INTO fact_check_sessions
              (session_id, raw_input, claim_results_json, final_report, status, error,
               active_run_id, created_at, updated_at)
            VALUES (?, ?, '[]', NULL, 'running', NULL, ?, ?, ?)
            """,
            (session_id, raw_input, run_id, now, now),
        )
        conn.execute(
            """
            INSERT INTO fact_check_runs
              (run_id, session_id, sequence, raw_input, claim_results_json,
               final_report, status, error, triggered_by, created_at, updated_at)
            VALUES (?, ?, 1, ?, '[]', NULL, 'running', NULL, 'initial', ?, ?)
            """,
            (run_id, session_id, raw_input, now, now),
        )
    return run_id


def create_factcheck_run(
    session_id: str,
    raw_input: str,
    triggered_by: str,
    db_path: Path | str | None = None,
) -> str:
    """Append a new fact-check run with status='running'."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    run_id = str(uuid.uuid4())
    now = time.time()
    with _get_connection(resolved) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM fact_check_runs
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        conn.execute(
            """
            INSERT INTO fact_check_runs
              (run_id, session_id, sequence, raw_input, claim_results_json,
               final_report, status, error, triggered_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, '[]', NULL, 'running', NULL, ?, ?, ?)
            """,
            (run_id, session_id, sequence, raw_input, triggered_by, now, now),
        )
    return run_id


def complete_factcheck_run(
    run_id: str,
    *,
    claim_results: list[dict[str, Any]],
    final_report: str | None,
    db_path: Path | str | None = None,
) -> None:
    """Mark a run done and sync legacy session payload columns."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    now = time.time()
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            UPDATE fact_check_runs
            SET claim_results_json = ?,
                final_report = ?,
                status = 'done',
                error = NULL,
                updated_at = ?
            WHERE run_id = ?
            """,
            (json.dumps(claim_results), final_report, now, run_id),
        )
        row = conn.execute(
            "SELECT * FROM fact_check_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            run = _parse_run_row(row)
            _sync_legacy_session_from_run(conn, run["session_id"], run)


def mark_factcheck_run_error(
    run_id: str,
    error: str,
    db_path: Path | str | None = None,
) -> None:
    """Mark a run as failed."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    now = time.time()
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            UPDATE fact_check_runs
            SET status = 'error', error = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (error, now, run_id),
        )


def set_active_run(
    session_id: str,
    run_id: str,
    db_path: Path | str | None = None,
) -> None:
    """Point the session at a run and sync legacy payload columns."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            UPDATE fact_check_sessions
            SET active_run_id = ?
            WHERE session_id = ?
            """,
            (run_id, session_id),
        )
        row = conn.execute(
            "SELECT * FROM fact_check_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            run = _parse_run_row(row)
            _sync_legacy_session_from_run(conn, session_id, run)


def get_factcheck_runs(
    session_id: str,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all runs for a session ordered by sequence."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        rows = conn.execute(
            """
            SELECT * FROM fact_check_runs
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()
    return [_parse_run_row(row) for row in rows]


def get_latest_run_sequence(
    session_id: str,
    db_path: Path | str | None = None,
) -> int:
    """Return the highest sequence among completed runs (0 if none)."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS latest
            FROM fact_check_runs
            WHERE session_id = ? AND status = 'done'
            """,
            (session_id,),
        ).fetchone()
    return int(row["latest"]) if row is not None else 0


def invalidate_fc_context(
    session_id: str,
    db_path: Path | str | None = None,
) -> None:
    """Remove cached dialogue fact-check context for a session."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        conn.execute(
            "DELETE FROM dialogue_fc_context WHERE session_id = ?",
            (session_id,),
        )


def get_initial_run_id(
    session_id: str,
    db_path: Path | str | None = None,
) -> str | None:
    """Return run_id for sequence=1, if present."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        row = conn.execute(
            """
            SELECT run_id FROM fact_check_runs
            WHERE session_id = ? AND sequence = 1
            """,
            (session_id,),
        ).fetchone()
    return row["run_id"] if row is not None else None


def update_session_status(
    session_id: str,
    status: str,
    *,
    error: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update session lifecycle status."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    now = time.time()
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            UPDATE fact_check_sessions
            SET status = ?, error = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (status, error, now, session_id),
        )


def try_acquire_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> bool:
    """Atomically transition done -> running. Returns True if acquired."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    now = time.time()
    with _get_connection(resolved) as conn:
        cursor = conn.execute(
            """
            UPDATE fact_check_sessions
            SET status = 'running', error = NULL, updated_at = ?
            WHERE session_id = ? AND status = 'done'
            """,
            (now, session_id),
        )
        return cursor.rowcount > 0


def get_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return session with active-run payload, run history, and messages."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        row = conn.execute(
            "SELECT * FROM fact_check_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None

        active_run_id = row["active_run_id"]
        active_run: dict[str, Any] | None = None
        if active_run_id:
            active_row = conn.execute(
                "SELECT * FROM fact_check_runs WHERE run_id = ?",
                (active_run_id,),
            ).fetchone()
            if active_row is not None:
                active_run = _parse_run_row(active_row)

        run_rows = conn.execute(
            """
            SELECT run_id, sequence, raw_input, status, triggered_by, created_at
            FROM fact_check_runs
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()

        message_rows = conn.execute(
            """
            SELECT role, content, timestamp AS created_at
            FROM dialogue_history
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
        ).fetchall()

    session = dict(row)
    if active_run is not None:
        session["raw_input"] = active_run["raw_input"]
        session["claim_results"] = active_run["claim_results"]
        session["final_report"] = active_run["final_report"]
    else:
        session["claim_results"] = json.loads(session.pop("claim_results_json"))
    session.pop("claim_results_json", None)
    session["active_run_id"] = active_run_id
    session["runs"] = [dict(r) for r in run_rows]
    session["messages"] = [dict(message) for message in message_rows]
    return session


def list_sessions(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return session summaries ordered by newest first."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        rows = conn.execute(
            """
            SELECT s.session_id,
                   COALESCE(r.raw_input, s.raw_input) AS raw_input,
                   s.status,
                   s.created_at,
                   s.updated_at
            FROM fact_check_sessions s
            LEFT JOIN fact_check_runs r
              ON r.session_id = s.session_id AND r.sequence = 1
            ORDER BY s.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str, db_path: Path | str | None = None) -> bool:
    """Delete a session and all related rows."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        for table in (
            "dialogue_history",
            "dialogue_summaries",
            "dialogue_fc_context",
            "fact_check_runs",
            "fact_check_sessions",
        ):
            conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        deleted = conn.total_changes > 0
    return deleted


def save_user_message(
    session_id: str,
    content: str,
    db_path: Path | str | None = None,
) -> None:
    """Persist a user message before a dialogue turn runs."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        conn.execute(
            """
            INSERT INTO dialogue_history
              (session_id, role, content, timestamp, intent, token_estimate)
            VALUES (?, 'user', ?, ?, NULL, 0)
            """,
            (session_id, content, time.time()),
        )


def session_exists(session_id: str, db_path: Path | str | None = None) -> bool:
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
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
    db_path: Path | str | None = None,
) -> None:
    """Back-compat shim: ensure session has one completed initial run."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    now = time.time()

    if not session_exists(session_id, db_path=resolved):
        with _get_connection(resolved) as conn:
            run_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO fact_check_sessions
                  (session_id, raw_input, claim_results_json, final_report, status,
                   error, active_run_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'done', NULL, ?, ?, ?)
                """,
                (
                    session_id,
                    raw_input,
                    json.dumps(claim_results),
                    final_report,
                    run_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO fact_check_runs
                  (run_id, session_id, sequence, raw_input, claim_results_json,
                   final_report, status, error, triggered_by, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?, ?, 'done', NULL, 'initial', ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    raw_input,
                    json.dumps(claim_results),
                    final_report,
                    now,
                    now,
                ),
            )
        return

    run_id = get_initial_run_id(session_id, db_path=resolved)
    if run_id is None:
        run_id = create_factcheck_run(
            session_id, raw_input, triggered_by="initial", db_path=resolved
        )
    complete_factcheck_run(
        run_id,
        claim_results=claim_results,
        final_report=final_report,
        db_path=resolved,
    )
    set_active_run(session_id, run_id, db_path=resolved)
    update_session_status(session_id, "done", db_path=resolved)


def load_session_for_dialogue(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load fact-check runs and dialogue state for a session."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    with _get_connection(resolved) as conn:
        fc_row = conn.execute(
            "SELECT * FROM fact_check_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if fc_row is None:
            raise KeyError(session_id)

        active_run_id = fc_row["active_run_id"]
        active_run: dict[str, Any] | None = None
        if active_run_id:
            active_row = conn.execute(
                "SELECT * FROM fact_check_runs WHERE run_id = ?",
                (active_run_id,),
            ).fetchone()
            if active_row is not None:
                active_run = _parse_run_row(active_row)

        completed_run_rows = conn.execute(
            """
            SELECT * FROM fact_check_runs
            WHERE session_id = ? AND status = 'done'
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()

        history_rows = conn.execute(
            "SELECT * FROM dialogue_history WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()

        summary_row = conn.execute(
            "SELECT * FROM dialogue_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        fc_context_row = conn.execute(
            """
            SELECT compressed_context, covers_through_sequence
            FROM dialogue_fc_context WHERE session_id = ?
            """,
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

    fact_check_runs = [_parse_run_row(row) for row in completed_run_rows]
    latest_run_sequence = max((r["sequence"] for r in fact_check_runs), default=0)

    covers_through_sequence: int | None = None
    compressed_fc_context: str | None = None
    if fc_context_row is not None:
        covers_through_sequence = fc_context_row["covers_through_sequence"]
        if (
            covers_through_sequence is not None
            and covers_through_sequence >= latest_run_sequence
        ):
            compressed_fc_context = fc_context_row["compressed_context"]

    if active_run is not None:
        raw_input = active_run["raw_input"]
        claim_results = active_run["claim_results"]
        final_report = active_run["final_report"]
    else:
        raw_input = fc_row["raw_input"]
        claim_results = json.loads(fc_row["claim_results_json"])
        final_report = fc_row["final_report"]

    return {
        "raw_input": raw_input,
        "claim_results": claim_results,
        "final_report": final_report,
        "fact_check_runs": fact_check_runs,
        "latest_run_sequence": latest_run_sequence,
        "fc_context_covers_sequence": covers_through_sequence,
        "dialogue_history": dialogue_history,
        "conversation_summary": conversation_summary,
        "compressed_fc_context": compressed_fc_context,
    }


def persist_dialogue_state(
    session_id: str,
    result: DialogueOutput,
    *,
    prior_history_len: int,
    db_path: Path | str | None = None,
) -> None:
    """Persist dialogue deltas after one turn."""
    resolved = _resolve_db_path(db_path)
    ensure_dialogue_tables(resolved)
    new_turns = result["dialogue_history"][prior_history_len:]

    with _get_connection(resolved) as conn:
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
        covers_sequence = result.get("fc_context_covers_sequence")
        if compressed and covers_sequence is not None:
            conn.execute(
                """
                INSERT INTO dialogue_fc_context
                  (session_id, compressed_context, created_at, covers_through_sequence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  compressed_context = excluded.compressed_context,
                  created_at = excluded.created_at,
                  covers_through_sequence = excluded.covers_through_sequence
                """,
                (session_id, compressed, time.time(), covers_sequence),
            )
