"""Conversations: the session a user works in, and who it belongs to.

This module owns the ``sessions`` table and one rule that the rest of the
system depends on: a session belongs to exactly one user, and any access from
another user is refused. Callers must go through :meth:`SessionStore.assert_owner`
before touching anything scoped to a session.

It does not execute analysis, store files or hold message content.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f-]{36}$")
DEFAULT_EMPLOYEE = "data_analyst"


class SessionError(ValueError):
    """Raised when a session cannot be created or read."""


class SessionAccessError(PermissionError):
    """Raised when a user reaches for a session that is not theirs."""


@dataclass(frozen=True)
class Session:
    session_id: str
    user_id: str
    employee: str
    title: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    """SQLite-backed store for sessions and their ownership."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    employee   TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions (user_id, created_at);
                """
            )

    def create_session(
        self,
        user_id: str,
        *,
        employee: str = DEFAULT_EMPLOYEE,
        title: str = "新的分析会话",
    ) -> Session:
        if not user_id:
            raise SessionError("创建会话必须提供用户。")

        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            employee=employee,
            title=title,
            created_at=_now(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions"
                " (session_id, user_id, employee, title, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    session.user_id,
                    session.employee,
                    session.title,
                    session.created_at,
                ),
            )
        return session

    def assert_owner(self, session_id: str, user_id: str) -> Session:
        """Return the session only if ``user_id`` owns it.

        A session that belongs to somebody else and a session that does not
        exist raise the same error with the same message on purpose: telling a
        caller "that session exists but is not yours" leaks whose it is.
        """

        if not SESSION_ID_PATTERN.match(session_id or ""):
            raise SessionAccessError("会话不存在或无权访问。")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None or row["user_id"] != user_id:
            raise SessionAccessError("会话不存在或无权访问。")
        return _row_to_session(row)

    def list_for_user(self, user_id: str, limit: int = 50) -> list[Session]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions WHERE user_id = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        return [_row_to_session(row) for row in rows]


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        user_id=row["user_id"],
        employee=row["employee"],
        title=row["title"],
        created_at=row["created_at"],
    )
