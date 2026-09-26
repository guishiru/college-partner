"""Runtime layout: where user data, state and generated files live.

This is not a business module. It exists so that every entry point (the web
app, the administrator CLI, the tests) resolves the same runtime root and the
same state database. Two entry points computing that path independently is how
the CLI ends up writing users into a database the web app never reads.

Nothing here knows about users, sessions, jobs or analysis.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

RUNTIME_ROOT_ENV = "DATA_ANALYST_RUNTIME_ROOT"
DEFAULT_ROOT_NAME = "codex-共建-runtime"
STATE_DIR_NAME = "state"
STATE_DATABASE_NAME = "workbench.sqlite"


def runtime_root(explicit: str | Path | None = None) -> Path:
    """Resolve the runtime root: explicit argument, environment, then default."""

    configured = os.environ.get(RUNTIME_ROOT_ENV)
    default_root = Path(tempfile.gettempdir()) / DEFAULT_ROOT_NAME
    return Path(explicit or configured or default_root)


def state_database_path(explicit_root: str | Path | None = None) -> Path:
    """Path of the SQLite file holding users, sessions and jobs."""

    return runtime_root(explicit_root) / STATE_DIR_NAME / STATE_DATABASE_NAME


def session_dir(explicit_root: str | Path | None, session_id: str) -> Path:
    """Directory holding one session's uploaded and generated files."""

    return runtime_root(explicit_root) / "sessions" / session_id
