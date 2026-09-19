"""Session-scoped file storage for the first web workflow."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


class FileStoreError(ValueError):
    """Raised when a file cannot be stored or resolved safely."""


class SessionFileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_upload(self, session_id: str, uploaded: FileStorage) -> dict[str, Any]:
        self._validate_session_id(session_id)
        original_name = uploaded.filename or ""
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise FileStoreError("仅支持 CSV、XLSX 和 XLS 文件。")

        safe_name = secure_filename(original_name) or f"upload{suffix}"
        file_id = uuid.uuid4().hex
        session_dir = self.root / "sessions" / session_id
        original_dir = session_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{file_id}_{safe_name}"
        path = original_dir / stored_name
        uploaded.save(path)

        manifest = {
            "file_id": file_id,
            "session_id": session_id,
            "original_name": original_name,
            "stored_name": stored_name,
            "path": str(path),
            "suffix": suffix,
        }
        self._write_manifest(session_dir, manifest)
        return manifest

    def resolve(self, session_id: str, file_id: str) -> dict[str, Any]:
        self._validate_session_id(session_id)
        if not re.fullmatch(r"[0-9a-f]{32}", file_id or ""):
            raise FileStoreError("文件标识无效。")
        manifest_path = self.root / "sessions" / session_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileStoreError("当前会话没有上传文件。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_id") != file_id:
            raise FileStoreError("文件不属于当前会话。")
        path = Path(manifest["path"]).resolve()
        session_root = (self.root / "sessions" / session_id).resolve()
        if session_root not in path.parents or not path.is_file():
            raise FileStoreError("文件路径无效或文件已不存在。")
        return manifest

    def _write_manifest(self, session_dir: Path, manifest: dict[str, Any]) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f-]{36}", session_id or ""):
            raise FileStoreError("会话标识无效。")
