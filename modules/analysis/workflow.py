"""Application workflow connecting data preparation, Skill planning, and engine.

Every entry point takes a :class:`~modules.jobs.JobContext`. That is deliberate:
the project conventions require every cross-module call to identify the user,
the session and the job, and the only reliable way to enforce that is to make
the call impossible to write without them. Results are stamped with the same
context so an analysis result can always be traced back to its origin.
"""

from __future__ import annotations

from typing import Any

from modules.analysis.data_processing import prepare_for_analysis
from modules.analysis.engine import AnalysisEngine, AnalysisEngineError
from modules.analysis.interpretation import explain_result
from modules.jobs import JobContext
from modules.skills.data_analyst import build_analysis_plan


class DataAnalystWorkflow:
    """Run the non-UI part of the first Data Analyst closed loop."""

    def __init__(self, engine: AnalysisEngine | None = None):
        self.engine = engine or AnalysisEngine()

    def inspect_and_plan(
        self,
        file_path: str,
        requirement: str,
        context: JobContext,
        client=None,
    ) -> dict[str, Any]:
        prepared = prepare_for_analysis(file_path)
        if not prepared.report.ready_for_analysis:
            return self._blocked(prepared, context)

        plan = build_analysis_plan(
            requirement,
            prepared.report.columns,
            prepared.report.column_types,
            client=client,
        )
        if not plan["can_execute"]:
            return self._needs_clarification(prepared, plan, context)

        return {
            "status": "ready",
            "message": "分析参数已齐全，请确认后执行。",
            "report": prepared.report,
            "plan": plan,
            "context": context.to_dict(),
        }

    def execute(
        self,
        file_path: str,
        requirement: str,
        context: JobContext,
        client=None,
    ) -> dict[str, Any]:
        prepared = prepare_for_analysis(file_path)
        if not prepared.report.ready_for_analysis:
            return self._blocked(prepared, context)

        plan = build_analysis_plan(
            requirement,
            prepared.report.columns,
            prepared.report.column_types,
            client=client,
        )
        if not plan["can_execute"]:
            return self._needs_clarification(prepared, plan, context)

        results = []
        errors = []
        for item in plan["methods"]:
            try:
                result = self.engine.run(
                    item["method"],
                    prepared.data,
                    item["params"],
                )
                result["interpretation"] = explain_result(result)
                result["provenance"] = self._provenance(context, file_path, result)
                results.append(result)
            except AnalysisEngineError as exc:
                errors.append(
                    {
                        "method": item["method"],
                        "label": item["label"],
                        "message": str(exc),
                    }
                )

        if not results and errors:
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "completed"
        return {
            "status": status,
            "message": "统计分析已完成。" if results else "统计分析未完成。",
            "report": prepared.report,
            "plan": plan,
            "results": results,
            "errors": errors,
            "context": context.to_dict(),
        }

    # -- shared shapes ---------------------------------------------------

    @staticmethod
    def _blocked(prepared, context: JobContext) -> dict[str, Any]:
        return {
            "status": "blocked",
            "message": "数据质量检查未通过，请先处理阻塞问题。",
            "report": prepared.report,
            "context": context.to_dict(),
        }

    @staticmethod
    def _needs_clarification(prepared, plan, context: JobContext) -> dict[str, Any]:
        return {
            "status": "need_clarification",
            "message": plan["question"],
            "report": prepared.report,
            "plan": plan,
            "context": context.to_dict(),
        }

    @staticmethod
    def _provenance(
        context: JobContext,
        file_path: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        meta = result.get("meta") or {}
        return {
            **context.to_dict(),
            "source_file": str(file_path),
            "method": result.get("method"),
            "method_version": meta.get("method_version"),
        }
