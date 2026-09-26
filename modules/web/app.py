"""Flask web application for the Data Analyst first closed loop.

This module is the composition root. It is the only place that knows about
users, sessions, jobs, files and the analysis workflow at the same time, and it
is where the three rules from the project conventions are enforced on every
request:

* the caller is resolved from a bearer token before anything else happens;
* the session must belong to that caller;
* the work runs inside a job, so it can be traced and retried without
  producing a second result.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from modules.analysis.workflow import DataAnalystWorkflow
from modules.conversations import SessionAccessError, SessionStore
from modules.employees import EmployeeError, list_employees, require_available
from modules.files import FileAccessError, FileStoreError, SessionFileStore
from modules.llm import LLMNotConfigured, client_from_environment
from modules.jobs import (
    JobAccessError,
    JobConflictError,
    JobContext,
    JobError,
    JobStore,
)
from modules.reports.service import ReportService
from modules.skills.dispatch import UnsupportedEmployee, writing_skill_for
from modules.users import AuthenticationError, UserStore
from modules.workspace import runtime_root, state_database_path

EMPLOYEE_DATA_ANALYST = "data_analyst"
SKILL_INSPECT = "data_analyst.inspect"
SKILL_ANALYZE = "data_analyst.analyze"


def create_app(root_dir: str | Path | None = None) -> Flask:
    root = runtime_root(root_dir)
    database = state_database_path(root)

    users = UserStore(database)
    sessions = SessionStore(database)
    jobs = JobStore(database)
    file_store = SessionFileStore(root)
    try:
        llm_client = client_from_environment()
    except LLMNotConfigured as exc:
        # 配置写错了要立刻知道，不能带着半截配置跑起来。
        raise RuntimeError(f"大模型配置有误：{exc}") from None
    workflow = DataAnalystWorkflow()
    report_service = ReportService()

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    app.extensions["workbench"] = {
        "root": root,
        "users": users,
        "sessions": sessions,
        "jobs": jobs,
        "files": file_store,
        "llm": llm_client,
    }

    # -- request helpers -------------------------------------------------

    def current_user():
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not token:
            token = (request.headers.get("X-Auth-Token") or "").strip()
        return users.resolve_token(token)

    def owned_session(user, session_id: str):
        return sessions.assert_owner((session_id or "").strip(), user.user_id)

    # -- open endpoints --------------------------------------------------

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "data-analyst"})

    @app.get("/api/methods")
    def methods():
        return jsonify({"methods": workflow.engine.list_methods()})

    # -- employees -------------------------------------------------------

    @app.get("/api/employees")
    def employees():
        """The sidebar and the workspace layout both come from here.

        Layout is an attribute of the employee, so the front end renders what
        the employee declares instead of assuming every employee wants the
        analyst's two-panel workspace.
        """

        return jsonify(
            {"employees": [employee.to_dict() for employee in list_employees()]}
        )

    # -- session lifecycle -----------------------------------------------

    @app.post("/api/sessions")
    def create_session():
        user = current_user()
        payload = request.get_json(silent=True) or {}
        title = (payload.get("title") or "新的分析会话").strip()[:120]
        employee = require_available(
            payload.get("employee") or EMPLOYEE_DATA_ANALYST
        )
        session = sessions.create_session(
            user.user_id,
            employee=employee.employee_id,
            title=title,
        )
        return jsonify(
            {
                "session_id": session.session_id,
                "title": session.title,
                "employee": employee.to_dict(),
                "username": user.username,
            }
        )

    @app.get("/api/sessions")
    def list_sessions():
        user = current_user()
        return jsonify(
            {
                "sessions": [
                    _serialize_value(session)
                    for session in sessions.list_for_user(user.user_id)
                ]
            }
        )

    def record_recognition_usage(job_id: str, payload: dict) -> None:
        """把这次识别花掉的 token 记到任务上。

        钱已经花了，所以无论识别成没成、任务最终成不成功，都要记账。
        """

        recognition = (payload or {}).get("recognition") or {}
        if not recognition.get("input_tokens") and not recognition.get("output_tokens"):
            return          # 关键词命中，没调模型
        jobs.record_usage(
            job_id,
            model=getattr(llm_client, "model", None) or getattr(llm_client, "name", None),
            input_tokens=int(recognition.get("input_tokens") or 0),
            output_tokens=int(recognition.get("output_tokens") or 0),
        )

    # -- analysis loop ---------------------------------------------------

    @app.post("/api/upload")
    def upload():
        user = current_user()
        session = owned_session(user, request.form.get("session_id"))
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "请选择要上传的文件。"}), 400

        manifest = file_store.save_upload(
            user.user_id, session.session_id, uploaded
        )
        job, _ = jobs.start(
            user_id=user.user_id,
            session_id=session.session_id,
            employee=EMPLOYEE_DATA_ANALYST,
            skill=SKILL_INSPECT,
            input_ref={"file_id": manifest["file_id"], "requirement": "查看数据结构"},
            idempotency_key=uuid.uuid4().hex,
        )
        context = JobContext(user.user_id, session.session_id, job.job_id)
        try:
            prepared = workflow.inspect_and_plan(
                manifest["path"], "查看数据结构", context, client=llm_client
            )
        except Exception as exc:
            jobs.fail(job.job_id, str(exc))
            raise
        record_recognition_usage(job.job_id, prepared)
        jobs.succeed(job.job_id, {"status": prepared["status"]})

        return jsonify(
            {
                "session_id": session.session_id,
                "job_id": job.job_id,
                "file_id": manifest["file_id"],
                "filename": manifest["original_name"],
                "report": _serialize_report(prepared["report"]),
            }
        )

    @app.post("/api/plan")
    def plan():
        user = current_user()
        payload = request.get_json(silent=True) or {}
        session = owned_session(user, payload.get("session_id"))
        file_id = payload.get("file_id")
        requirement = (payload.get("requirement") or "").strip()
        if not file_id or not requirement:
            return jsonify({"error": "请提供文件和分析需求。"}), 400

        manifest = file_store.resolve(user.user_id, session.session_id, file_id)
        job, _ = jobs.start(
            user_id=user.user_id,
            session_id=session.session_id,
            employee=EMPLOYEE_DATA_ANALYST,
            skill=SKILL_INSPECT,
            input_ref={"file_id": file_id, "requirement": requirement},
            idempotency_key=uuid.uuid4().hex,
        )
        context = JobContext(user.user_id, session.session_id, job.job_id)
        try:
            result = workflow.inspect_and_plan(
                manifest["path"], requirement, context, client=llm_client
            )
        except Exception as exc:
            jobs.fail(job.job_id, str(exc))
            raise
        record_recognition_usage(job.job_id, result)
        jobs.succeed(job.job_id, {"status": result["status"]})
        return jsonify(_serialize_payload(result))

    @app.post("/api/execute")
    def execute():
        user = current_user()
        payload = request.get_json(silent=True) or {}
        session = owned_session(user, payload.get("session_id"))
        file_id = payload.get("file_id")
        requirement = (payload.get("requirement") or "").strip()
        if not file_id or not requirement:
            return jsonify({"error": "请提供文件和分析需求。"}), 400

        manifest = file_store.resolve(user.user_id, session.session_id, file_id)
        job, reused = jobs.start(
            user_id=user.user_id,
            session_id=session.session_id,
            employee=EMPLOYEE_DATA_ANALYST,
            skill=SKILL_ANALYZE,
            input_ref={"file_id": file_id, "requirement": requirement},
            idempotency_key=_analysis_key(file_id, requirement),
        )

        if reused:
            stored = _read_stored_result(file_store, session.session_id, job)
            if stored is not None:
                stored["reused"] = True
                stored["job_id"] = job.job_id
                return jsonify(stored)

        context = JobContext(user.user_id, session.session_id, job.job_id)
        try:
            result = workflow.execute(
                manifest["path"], requirement, context, client=llm_client
            )
            record_recognition_usage(job.job_id, result)
            response = _serialize_payload(result)
            response["job_id"] = job.job_id
            response["reused"] = False

            if result.get("status") in {"completed", "partial"} and result.get("results"):
                report_id = uuid.uuid4().hex
                report_path = file_store.report_path(session.session_id, report_id)
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
                    f"/api/reports/{session.session_id}/{report_id}/download"
                )
        except Exception as exc:
            jobs.fail(job.job_id, str(exc))
            raise

        result_path = file_store.result_path(session.session_id, job.job_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8"
        )
        jobs.succeed(
            job.job_id,
            {
                "status": result["status"],
                "result_file": str(result_path),
                "report_id": response.get("report_id"),
                "download_url": response.get("download_url"),
            },
        )
        return jsonify(response)

    @app.get("/api/files/<session_id>")
    def list_files(session_id: str):
        user = current_user()
        session = owned_session(user, session_id)
        return jsonify(
            {"files": file_store.list_files(user.user_id, session.session_id)}
        )

    @app.get("/api/jobs/<session_id>")
    def list_jobs(session_id: str):
        user = current_user()
        session = owned_session(user, session_id)
        return jsonify(
            {
                "jobs": [
                    _serialize_value(job)
                    for job in jobs.list_for_session(session.session_id, user.user_id)
                ]
            }
        )

    @app.get("/api/jobs/<session_id>/<job_id>/result")
    def job_result(session_id: str, job_id: str):
        """Read back a finished analysis.

        Without this, everything but the newest result in a session is
        unreachable from the interface — the report file and the job record are
        both on disk, but nothing can open them again.
        """

        user = current_user()
        session = owned_session(user, session_id)
        job = jobs.get_for_user(job_id, user.user_id)
        if job.session_id != session.session_id:
            raise JobAccessError("任务不存在或无权访问。")
        stored = _read_stored_result(file_store, session.session_id, job)
        if stored is None:
            return jsonify({"error": "该任务没有可回看的结果。"}), 404
        stored["job_id"] = job.job_id
        stored["reused"] = True
        return jsonify(stored)

    # -- 写作类员工 -------------------------------------------------------

    def writing_context(payload):
        """写作接口共用的前置：鉴权、会话归属、员工是否提供写作能力。"""

        user = current_user()
        session = owned_session(user, payload.get("session_id"))
        skill = writing_skill_for(session.employee)
        brief = (payload.get("brief") or "").strip()
        if not brief:
            raise ValueError("请说明要写什么。")
        return user, session, skill, brief

    def run_writing(job, result):
        """记账 + 组装响应。

        用量无论成稿与否都要记——重写了三次没通过，钱一样花掉了。
        """

        if result.usage.calls:
            jobs.record_usage(
                job.job_id,
                model=getattr(llm_client, "model", None) or getattr(llm_client, "name", None),
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                calls=result.usage.calls,
            )
        payload = {
            "job_id": job.job_id,
            "text": result.text,
            "ready_to_deliver": result.ready_to_deliver,
            "rewrites": result.rewrites,
            "failed": result.failed,
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "message": f.message,
                    "location": f.location,
                }
                for f in result.review.findings
            ],
            "usage": {
                "calls": result.usage.calls,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
        }
        if result.failed:
            jobs.fail(job.job_id, result.failed)
        else:
            jobs.succeed(job.job_id, {
                "ready_to_deliver": result.ready_to_deliver,
                "rewrites": result.rewrites,
            })
        return payload

    @app.post("/api/documents/skeleton")
    def document_skeleton():
        """第一轮只出骨架，等用户确认。爆炸点几乎总在用户看到骨架那一刻。"""

        payload = request.get_json(silent=True) or {}
        user, session, skill, brief = writing_context(payload)
        job, _ = jobs.start(
            user_id=user.user_id,
            session_id=session.session_id,
            employee=session.employee,
            skill=skill.skeleton_skill,
            input_ref={"brief": brief},
            idempotency_key=uuid.uuid4().hex,
        )
        result = skill.skeleton(brief, client=llm_client)
        return jsonify(run_writing(job, result))

    @app.post("/api/documents/draft")
    def document_draft():
        """用户确认骨架后成稿，过校验回路。"""

        payload = request.get_json(silent=True) or {}
        user, session, skill, brief = writing_context(payload)
        skeleton = (payload.get("skeleton") or "").strip()
        full_brief = f"{brief}\n\n已确认的骨架：\n{skeleton}" if skeleton else brief

        job, reused = jobs.start(
            user_id=user.user_id,
            session_id=session.session_id,
            employee=session.employee,
            skill=skill.draft_skill,
            input_ref={"brief": brief, "skeleton": skeleton},
            idempotency_key=_analysis_key(session.employee, full_brief),
        )
        if reused:
            stored = _read_stored_result(file_store, session.session_id, job)
            if stored is not None:
                stored["reused"] = True
                return jsonify(stored)

        result = skill.draft(full_brief, client=llm_client)
        response = run_writing(job, result)
        response["reused"] = False

        if result.text:
            path = file_store.result_path(session.session_id, job.job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        return jsonify(response)

    @app.post("/api/documents/review")
    def document_review():
        """只审不改：把已有稿子过一遍校验器，不调模型。"""

        payload = request.get_json(silent=True) or {}
        user = current_user()
        session = owned_session(user, payload.get("session_id"))
        skill = writing_skill_for(session.employee)
        text = payload.get("text") or ""
        if not text.strip():
            return jsonify({"error": "请提供要审阅的文稿。"}), 400
        review = skill.review(text)
        return jsonify({
            "ready_to_deliver": review.ready_to_deliver,
            "findings": [
                {"code": f.code, "severity": f.severity,
                 "message": f.message, "location": f.location}
                for f in review.findings
            ],
        })

    @app.get("/api/usage")
    def usage_for_user():
        """当前用户的累计用量。做限额时从这里读。"""

        user = current_user()
        return jsonify(jobs.usage_for_user(user.user_id))

    @app.get("/api/usage/<session_id>")
    def usage_for_session(session_id: str):
        user = current_user()
        session = owned_session(user, session_id)
        return jsonify(jobs.usage_for_session(session.session_id, user.user_id))

    @app.get("/api/reports/<session_id>/<report_id>/download")
    def download_report(session_id: str, report_id: str):
        user = current_user()
        session = owned_session(user, session_id)
        path = file_store.report_path(session.session_id, report_id)
        if not path.is_file():
            return jsonify({"error": "报告不存在或已过期。"}), 404
        return send_file(
            path,
            as_attachment=True,
            download_name="数据分析报告.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    # -- error handling --------------------------------------------------

    @app.errorhandler(AuthenticationError)
    def _unauthenticated(exc):
        return jsonify({"error": str(exc)}), 401

    @app.errorhandler(SessionAccessError)
    @app.errorhandler(FileAccessError)
    @app.errorhandler(JobAccessError)
    def _forbidden(exc):
        return jsonify({"error": str(exc)}), 403

    @app.errorhandler(JobConflictError)
    def _conflict(exc):
        return jsonify({"error": str(exc)}), 409

    @app.errorhandler(UnsupportedEmployee)
    @app.errorhandler(EmployeeError)
    @app.errorhandler(FileStoreError)
    @app.errorhandler(JobError)
    @app.errorhandler(ValueError)
    def _bad_request(exc):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(Exception)
    def _unexpected(exc):
        app.logger.exception("请求处理失败")
        return jsonify({"error": f"请求处理失败：{exc}"}), 500

    return app


def _analysis_key(file_id: str, requirement: str) -> str:
    """Same file plus same requirement is the same analysis."""

    digest = hashlib.sha256(f"{file_id}|{requirement}".encode("utf-8"))
    return digest.hexdigest()


def _read_stored_result(file_store, session_id: str, job) -> dict[str, Any] | None:
    reference = job.result_ref or {}
    stored = reference.get("result_file")
    path = Path(stored) if stored else file_store.result_path(session_id, job.job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host=host, port=port, debug=False)
