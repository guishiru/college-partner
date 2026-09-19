"""Flask web application for the Data Analyst first closed loop."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from modules.analysis.workflow import DataAnalystWorkflow
from modules.files.service import FileStoreError, SessionFileStore
from modules.reports.service import ReportService


def create_app(runtime_root: str | Path | None = None) -> Flask:
    configured_root = os.environ.get("DATA_ANALYST_RUNTIME_ROOT")
    default_root = Path(tempfile.gettempdir()) / "codex-共建-runtime"
    root = Path(runtime_root or configured_root or default_root)
    file_store = SessionFileStore(root)
    workflow = DataAnalystWorkflow()
    report_service = ReportService()

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "data-analyst"})

    @app.get("/api/methods")
    def methods():
        return jsonify({"methods": workflow.engine.list_methods()})

    @app.post("/api/upload")
    def upload():
        session_id = (request.form.get("session_id") or "").strip()
        uploaded = request.files.get("file")
        if not session_id or not uploaded or not uploaded.filename:
            return jsonify({"error": "请选择文件并提供有效会话。"}), 400
        try:
            manifest = file_store.save_upload(session_id, uploaded)
            prepared = workflow.inspect_and_plan(
                manifest["path"],
                "查看数据结构",
            )
            return jsonify(
                {
                    "session_id": session_id,
                    "file_id": manifest["file_id"],
                    "filename": manifest["original_name"],
                    "report": _serialize_report(prepared["report"]),
                }
            )
        except (FileStoreError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"文件处理失败：{exc}"}), 500

    @app.post("/api/plan")
    def plan():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        file_id = payload.get("file_id")
        requirement = (payload.get("requirement") or "").strip()
        if not session_id or not file_id or not requirement:
            return jsonify({"error": "请提供文件、会话和分析需求。"}), 400
        try:
            manifest = file_store.resolve(session_id, file_id)
            result = workflow.inspect_and_plan(manifest["path"], requirement)
            return jsonify(_serialize_payload(result))
        except (FileStoreError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"需求解析失败：{exc}"}), 500

    @app.post("/api/execute")
    def execute():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        file_id = payload.get("file_id")
        requirement = (payload.get("requirement") or "").strip()
        if not session_id or not file_id or not requirement:
            return jsonify({"error": "请提供文件、会话和分析需求。"}), 400
        try:
            manifest = file_store.resolve(session_id, file_id)
            result = workflow.execute(manifest["path"], requirement)
            response = _serialize_payload(result)
            if result.get("status") in {"completed", "partial"} and result.get("results"):
                report_id = uuid.uuid4().hex
                report_path = (
                    root / "sessions" / session_id / "reports" / f"{report_id}.docx"
                )
                report_service.build_word_report(
                    report_path,
                    filename=manifest["original_name"],
                    quality_report=result["report"].to_dict(),
                    plan=result["plan"],
                    results=result["results"],
                    errors=result.get("errors") or [],
                )
                response["report_id"] = report_id
                response["download_url"] = (
                    f"/api/reports/{session_id}/{report_id}/download"
                )
            return jsonify(response)
        except (FileStoreError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"分析执行失败：{exc}"}), 500

    @app.get("/api/reports/<session_id>/<report_id>/download")
    def download_report(session_id: str, report_id: str):
        if not _is_uuid_like(session_id) or not _is_hex_id(report_id):
            return jsonify({"error": "报告标识无效。"}), 400
        path = root / "sessions" / session_id / "reports" / f"{report_id}.docx"
        if not path.is_file():
            return jsonify({"error": "报告不存在或已过期。"}), 404
        return send_file(
            path,
            as_attachment=True,
            download_name="数据分析报告.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    return app


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in payload.items()}


def _serialize_report(report) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return _serialize_value(report.to_dict())
    return _serialize_value(report)


def _serialize_value(value):
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "rows": [_serialize_value(row) for row in value.to_dict(orient="records")],
        }
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _is_uuid_like(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _is_hex_id(value: str) -> bool:
    return len(value or "") == 32 and all(
        character in "0123456789abcdef" for character in value
    )


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host=host, port=port, debug=False)
