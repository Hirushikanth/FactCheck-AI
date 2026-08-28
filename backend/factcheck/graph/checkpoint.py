"""Lifecycle-owned SQLite checkpointing for LangGraph state graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from factcheck.config import BACKEND_DIR

logger = logging.getLogger(__name__)

try:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover - dependency is installed in deployed environments
    aiosqlite = None  # type: ignore[assignment]
    AsyncSqliteSaver = None  # type: ignore[assignment,misc]


def thread_config(session_id: str) -> dict[str, dict[str, str]]:
    """Return the LangGraph configuration for one browser-owned session."""

    return {"configurable": {"thread_id": session_id}}


def resolve_checkpoint_path(db_path: Path | str) -> str:
    """Resolve checkpoint storage beside the configured session SQLite database."""

    path = Path(db_path)
    return str(path if path.is_absolute() else BACKEND_DIR / path)


class CheckpointManager:
    """Own one AsyncSqliteSaver and connection for the application lifetime."""

    def __init__(self) -> None:
        self._connection: Any = None
        self._saver: Any = None

    @property
    def saver(self) -> Any:
        return self._saver

    async def start(self, db_path: Path | str) -> Any:
        """Open SQLite and initialize the LangGraph checkpoint tables."""

        if self._saver is not None:
            return self._saver
        if aiosqlite is None or AsyncSqliteSaver is None:
            logger.error(
                "SQLite LangGraph checkpointing is unavailable; install "
                "langgraph-checkpoint-sqlite and aiosqlite before graph workflows."
            )
            return None

        self._connection = await aiosqlite.connect(resolve_checkpoint_path(db_path))
        self._saver = AsyncSqliteSaver(self._connection)
        await self._saver.setup()
        return self._saver

    async def stop(self) -> None:
        """Close the shared SQLite connection during application shutdown."""

        connection, self._connection = self._connection, None
        self._saver = None
        if connection is not None:
            await connection.close()


checkpoint_manager = CheckpointManager()


def get_checkpointer() -> Any:
    """Return the active saver, or ``None`` for dependency-less unit tests."""

    return checkpoint_manager.saver


async def startup_checkpointer(db_path: Path | str) -> Any:
    return await checkpoint_manager.start(db_path)


async def shutdown_checkpointer() -> None:
    await checkpoint_manager.stop()

