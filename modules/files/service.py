"""Session-scoped file storage for the first web workflow.

Every stored file records the user, the session and — for generated files —
the job that produced it, so a file can always be traced back to its owner.
Ownership of the *session* is decided by the conversations module; this module
refuses any file whose recorded ``user_id`` does not match the caller.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

USER_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SESSION_ID_PATTERN = re.compile(r"^[0-9a-f-]{36}$")
FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}


class FileStoreError(ValueError):
    """Raised when a file cannot be stored or resolved safely."""


class FileAccessError(PermissionError):
    """Raised when a file is reached for by someone who does not own it."""


class SessionFileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        user_id: str,
        session_id: str,
        uploaded: FileStorage,
    ) -> dict[str, Any]:
        self._validate_user_id(user_id)
        self._validate_session_id(session_id)
        original_name = uploaded.filename or ""
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise FileStoreError("仅支持 CSV、XLSX 和 XLS 文件。")

        # ``secure_filename`` drops non-ASCII characters, so "一月.csv" comes back
        # as "csv" — the extension becomes the whole name and the stored file
        # ends up with none. Sanitise the stem only and re-attach the suffix we
        # already validated above. The user-facing name is kept in the manifest.
        safe_stem = secure_filename(Path(original_name).stem) or "upload"
        safe_name = f"{safe_stem}{suffix}"
        file_id = uuid.uuid4().hex
        session_dir = self.root / "sessions" / session_id
        original_dir = session_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{file_id}_{safe_name}"
        path = original_dir / stored_name
        uploaded.save(path)

        manifest = {
            "file_id": file_id,
            "user_id": user_id,
            "session_id": session_id,
            "original_name": original_name,
            "stored_name": stored_name,
            "path": str(path),
            "suffix": suffix,
            # Microseconds, not seconds: two uploads in the same second would
            # tie and the "newest first" listing would flip order between runs.
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        self._write_manifest(session_id, manifest)
        return manifest

    def resolve(self, user_id: str, session_id: str, file_id: str) -> dict[str, Any]:
        self._validate_user_id(user_id)
        self._validate_session_id(session_id)
        if not FILE_ID_PATTERN.match(file_id or ""):
            raise FileStoreError("文件标识无效。")

        manifest_path = self._manifest_path(session_id, file_id)
        if not manifest_path.is_file():
            raise FileStoreError("文件不属于当前会话。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("user_id") != user_id:
            raise FileAccessError("文件不存在或无权访问。")

        path = Path(manifest["path"]).resolve()
        session_root = (self.root / "sessions" / session_id).resolve()
        if session_root not in path.parents or not path.is_file():
            raise FileStoreError("文件路径无效或文件已不存在。")
        return manifest

    def list_files(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Every file this user uploaded into the session, newest first."""

        self._validate_user_id(user_id)
        self._validate_session_id(session_id)
        manifests = []
        for path in self._manifest_dir(session_id).glob("*.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("user_id") == user_id:
                manifests.append(manifest)
        manifests.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
        return manifests

    # -- generated files -------------------------------------------------

    def report_path(self, session_id: str, report_id: str) -> Path:
        """Path of a generated report.

        Session ownership is the conversations module's rule; the caller must
        have asserted it before calling this. Here we only guarantee that the
        resulting path cannot escape the session directory.
        """

        self._validate_session_id(session_id)
        if not FILE_ID_PATTERN.match(report_id or ""):
            raise FileStoreError("报告标识无效。")
        return self.root / "sessions" / session_id / "reports" / f"{report_id}.docx"

    def result_path(self, session_id: str, job_id: str) -> Path:
        """Path of a job's stored analysis payload.

        Keeping the payload on disk and only a reference in the job record is
        what lets a retried job return its original result instead of running
        the analysis a second time.
        """

        self._validate_session_id(session_id)
        if not FILE_ID_PATTERN.match(job_id or ""):
            raise FileStoreError("任务标识无效。")
        return self.root / "sessions" / session_id / "results" / f"{job_id}.json"

    # -- helpers ---------------------------------------------------------

    def _manifest_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id / "manifests"

    def _manifest_path(self, session_id: str, file_id: str) -> Path:
        return self._manifest_dir(session_id) / f"{file_id}.json"

    def _write_manifest(self, session_id: str, manifest: dict[str, Any]) -> None:
        """One manifest file per upload.

        An earlier version kept a single ``manifest.json`` per session, so a
        second upload overwrote the first one's record and left its bytes on
        disk with nothing pointing at them. Keying the manifest by ``file_id``
        is what makes several files per session possible at all.
        """

        directory = self._manifest_dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path(session_id, manifest["file_id"]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not USER_ID_PATTERN.match(user_id or ""):
            raise FileStoreError("用户标识无效。")

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not SESSION_ID_PATTERN.match(session_id or ""):
            raise FileStoreError("会话标识无效。")
