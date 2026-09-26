"""Jobs: one trackable unit of work, and the retry rule that guards it.

This module owns the ``jobs`` table. A job records who asked, in which session,
which virtual employee and Skill were used, what went in, what came out and why
it failed.

It implements one rule from the project conventions that cannot be expressed in
prose alone:

    任务失败后允许重试，但不能重复生成结果。

:meth:`JobStore.start` enforces it with a unique key per
``(session_id, idempotency_key)``:

* a **succeeded** job is returned as-is and the caller must reuse its result
  instead of recomputing;
* a **running** job is refused, so two concurrent requests cannot both produce
  a result;
* a **failed** job is moved back to running and its attempt counter advances.

The module never runs analysis and never reads another module's tables.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


class JobError(ValueError):
    """Raised when a job cannot be created or read."""


class JobAccessError(PermissionError):
    """Raised when a user reaches for a job that is not theirs."""


class JobConflictError(RuntimeError):
    """Raised when the same job is already running."""


@dataclass(frozen=True)
class Job:
    job_id: str
    user_id: str
    session_id: str
    employee: str
    skill: str
    input_ref: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_RUNNING
    result_ref: dict[str, Any] | None = None
    failure_reason: str | None = None
    attempts: int = 1
    #: 模型用量。一个任务可能调多次模型，这几项是累计值。
    model: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class JobContext:
    """The identity every cross-module call must carry."""

    user_id: str
    session_id: str
    job_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "job_id": self.job_id,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """SQLite-backed store for jobs and their lifecycle."""

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
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id          TEXT PRIMARY KEY,
                    user_id         TEXT NOT NULL,
                    session_id      TEXT NOT NULL,
                    employee        TEXT NOT NULL,
                    skill           TEXT NOT NULL,
                    input_ref       TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    result_ref      TEXT,
                    failure_reason  TEXT,
                    attempts        INTEGER NOT NULL DEFAULT 1,
                    model           TEXT,
                    llm_calls       INTEGER NOT NULL DEFAULT 0,
                    input_tokens    INTEGER NOT NULL DEFAULT 0,
                    output_tokens   INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
                    ON jobs (session_id, idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_jobs_session
                    ON jobs (session_id, created_at);
                """
            )

    def start(
        self,
        *,
        user_id: str,
        session_id: str,
        employee: str,
        skill: str,
        input_ref: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[Job, bool]:
        """Claim a job slot.

        Returns ``(job, reused)``. When ``reused`` is true the job already
        succeeded and the caller must return its stored result rather than
        running the work again.
        """

        if not user_id or not session_id:
            raise JobError("任务必须绑定用户和会话。")
        if not idempotency_key:
            raise JobError("任务必须提供幂等键。")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE session_id = ? AND idempotency_key = ?",
                (session_id, idempotency_key),
            ).fetchone()

            if row is not None:
                if row["user_id"] != user_id:
                    connection.execute("ROLLBACK")
                    raise JobAccessError("任务不存在或无权访问。")
                if row["status"] == STATUS_SUCCEEDED:
                    connection.execute("COMMIT")
                    return _row_to_job(row), True
                if row["status"] == STATUS_RUNNING:
                    connection.execute("ROLLBACK")
                    raise JobConflictError("同一任务正在执行中，请等待完成后再试。")

                connection.execute(
                    "UPDATE jobs SET status = ?, failure_reason = NULL,"
                    " attempts = attempts + 1, updated_at = ?"
                    " WHERE job_id = ?",
                    (STATUS_RUNNING, _now(), row["job_id"]),
                )
                refreshed = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
                ).fetchone()
                connection.execute("COMMIT")
                return _row_to_job(refreshed), False

            timestamp = _now()
            job = Job(
                job_id=uuid.uuid4().hex,
                user_id=user_id,
                session_id=session_id,
                employee=employee,
                skill=skill,
                input_ref=dict(input_ref or {}),
                status=STATUS_RUNNING,
                attempts=1,
                created_at=timestamp,
                updated_at=timestamp,
            )
            connection.execute(
                "INSERT INTO jobs (job_id, user_id, session_id, employee, skill,"
                " input_ref, status, attempts, idempotency_key, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.user_id,
                    job.session_id,
                    job.employee,
                    job.skill,
                    json.dumps(job.input_ref, ensure_ascii=False),
                    job.status,
                    job.attempts,
                    idempotency_key,
                    job.created_at,
                    job.updated_at,
                ),
            )
            connection.execute("COMMIT")
            return job, False
        except (JobAccessError, JobConflictError):
            raise
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def record_usage(
        self,
        job_id: str,
        *,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        calls: int = 1,
    ) -> Job:
        """累加一次模型调用的用量。

        必须是累加而不是覆盖：一个任务可能调多次模型——识别一次、成稿一次、
        校验没过再重写一次。覆盖的话账就永远少算。

        用量是记账，不是业务结果：即便任务最终失败，这几次调用的钱也已经花掉
        了，所以不限制任务状态。
        """

        if input_tokens < 0 or output_tokens < 0 or calls < 0:
            raise JobError("用量不能为负数。")

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET"
                " llm_calls = llm_calls + ?,"
                " input_tokens = input_tokens + ?,"
                " output_tokens = output_tokens + ?,"
                " model = COALESCE(?, model),"
                " updated_at = ?"
                " WHERE job_id = ?",
                (calls, int(input_tokens), int(output_tokens), model, _now(), job_id),
            )
            if cursor.rowcount == 0:
                raise JobError("任务不存在。")
        return self.get(job_id)

    def usage_for_session(self, session_id: str, user_id: str) -> dict[str, Any]:
        """一个会话花了多少。"""

        return self._usage(
            "WHERE session_id = ? AND user_id = ?", (session_id, user_id)
        )

    def usage_for_user(self, user_id: str) -> dict[str, Any]:
        """一个用户总共花了多少。做限额的话从这里读。"""

        return self._usage("WHERE user_id = ?", (user_id,))

    def _usage(self, where: str, params: tuple) -> dict[str, Any]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS jobs,"
                " COALESCE(SUM(llm_calls), 0) AS llm_calls,"
                " COALESCE(SUM(input_tokens), 0) AS input_tokens,"
                " COALESCE(SUM(output_tokens), 0) AS output_tokens"
                f" FROM jobs {where}",
                params,
            ).fetchone()
            per_skill = connection.execute(
                "SELECT skill, employee, COUNT(*) AS jobs,"
                " COALESCE(SUM(llm_calls), 0) AS llm_calls,"
                " COALESCE(SUM(input_tokens), 0) AS input_tokens,"
                " COALESCE(SUM(output_tokens), 0) AS output_tokens"
                f" FROM jobs {where} GROUP BY skill, employee"
                " ORDER BY input_tokens + output_tokens DESC",
                params,
            ).fetchall()

        return {
            "jobs": int(total["jobs"]),
            "llm_calls": int(total["llm_calls"]),
            "input_tokens": int(total["input_tokens"]),
            "output_tokens": int(total["output_tokens"]),
            "total_tokens": int(total["input_tokens"]) + int(total["output_tokens"]),
            "by_skill": [
                {
                    "skill": row["skill"],
                    "employee": row["employee"],
                    "jobs": int(row["jobs"]),
                    "llm_calls": int(row["llm_calls"]),
                    "input_tokens": int(row["input_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "total_tokens": int(row["input_tokens"]) + int(row["output_tokens"]),
                }
                for row in per_skill
            ],
        }

    def succeed(self, job_id: str, result_ref: dict[str, Any]) -> Job:
        return self._finish(job_id, STATUS_SUCCEEDED, result_ref, None)

    def fail(self, job_id: str, reason: str) -> Job:
        return self._finish(job_id, STATUS_FAILED, None, str(reason))

    def _finish(
        self,
        job_id: str,
        status: str,
        result_ref: dict[str, Any] | None,
        failure_reason: str | None,
    ) -> Job:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, result_ref = ?, failure_reason = ?,"
                " updated_at = ? WHERE job_id = ? AND status = ?",
                (
                    status,
                    json.dumps(result_ref, ensure_ascii=False) if result_ref else None,
                    failure_reason,
                    _now(),
                    job_id,
                    STATUS_RUNNING,
                ),
            )
            if cursor.rowcount == 0:
                raise JobError("任务不存在或不处于执行中状态。")
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobError("任务不存在。")
        return _row_to_job(row)

    def get_for_user(self, job_id: str, user_id: str) -> Job:
        job = self.get(job_id)
        if job.user_id != user_id:
            raise JobAccessError("任务不存在或无权访问。")
        return job

    def list_for_session(self, session_id: str, user_id: str) -> list[Job]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE session_id = ? AND user_id = ?"
                " ORDER BY created_at DESC",
                (session_id, user_id),
            ).fetchall()
        return [_row_to_job(row) for row in rows]


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        employee=row["employee"],
        skill=row["skill"],
        input_ref=json.loads(row["input_ref"]) if row["input_ref"] else {},
        status=row["status"],
        result_ref=json.loads(row["result_ref"]) if row["result_ref"] else None,
        failure_reason=row["failure_reason"],
        attempts=row["attempts"],
        model=row["model"],
        llm_calls=row["llm_calls"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
