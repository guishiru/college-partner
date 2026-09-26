"""User identity, status and access tokens.

This module owns the ``users`` and ``user_tokens`` tables. No other module may
read those tables directly. Callers resolve a token into a :class:`User` and
then carry ``user_id`` across module boundaries.

Tokens are issued out of band by an administrator (see ``modules.users.cli``)
and are stored only as SHA-256 digests, so the database never holds a usable
credential.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
USERNAME_PATTERN = re.compile(r"^[0-9A-Za-z_.@-]{3,64}$")


class UserError(ValueError):
    """Raised when a user cannot be created or read."""


class AuthenticationError(PermissionError):
    """Raised when a token does not resolve to an active user."""


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    status: str
    created_at: str

    @property
    def is_active(self) -> bool:
        return self.status == STATUS_ACTIVE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


class UserStore:
    """SQLite-backed store for users and their access tokens."""

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
                CREATE TABLE IF NOT EXISTS users (
                    user_id    TEXT PRIMARY KEY,
                    username   TEXT NOT NULL UNIQUE,
                    status     TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_user_tokens_user
                    ON user_tokens (user_id);
                """
            )

    # -- users -----------------------------------------------------------

    def create_user(self, username: str) -> User:
        username = (username or "").strip()
        if not USERNAME_PATTERN.match(username):
            raise UserError("用户名只允许字母、数字、下划线、点、@ 和连字符，长度 3-64。")

        user = User(
            user_id=uuid.uuid4().hex,
            username=username,
            status=STATUS_ACTIVE,
            created_at=_now(),
        )
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO users (user_id, username, status, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (user.user_id, user.username, user.status, user.created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise UserError(f"用户名已存在：{username}") from exc
        return user

    def get_user(self, user_id: str) -> User:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise UserError("用户不存在。")
        return _row_to_user(row)

    def set_status(self, user_id: str, status: str) -> User:
        if status not in {STATUS_ACTIVE, STATUS_DISABLED}:
            raise UserError(f"不支持的用户状态：{status}")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET status = ? WHERE user_id = ?", (status, user_id)
            )
            if cursor.rowcount == 0:
                raise UserError("用户不存在。")
        return self.get_user(user_id)

    # -- tokens ----------------------------------------------------------

    def issue_token(self, user_id: str) -> str:
        """Create an access token. The plaintext is returned exactly once."""

        user = self.get_user(user_id)
        if not user.is_active:
            raise UserError("已停用的用户不能签发令牌。")

        token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO user_tokens (token_hash, user_id, created_at)"
                " VALUES (?, ?, ?)",
                (_digest(token), user_id, _now()),
            )
        return token

    def revoke_token(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE user_tokens SET revoked_at = ?"
                " WHERE token_hash = ? AND revoked_at IS NULL",
                (_now(), _digest(token)),
            )

    def resolve_token(self, token: str) -> User:
        """Resolve a bearer token into an active user, or raise."""

        if not token:
            raise AuthenticationError("缺少访问令牌。")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT u.* FROM user_tokens t"
                " JOIN users u ON u.user_id = t.user_id"
                " WHERE t.token_hash = ? AND t.revoked_at IS NULL",
                (_digest(token),),
            ).fetchone()
        if row is None:
            raise AuthenticationError("访问令牌无效或已撤销。")
        user = _row_to_user(row)
        if not user.is_active:
            raise AuthenticationError("用户已停用。")
        return user


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        user_id=row["user_id"],
        username=row["username"],
        status=row["status"],
        created_at=row["created_at"],
    )
