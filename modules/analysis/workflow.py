"""Application workflow connecting data preparation, Skill planning, and engine."""

from __future__ import annotations

from typing import Any

from modules.analysis.data_processing import prepare_for_analysis
from modules.analysis.engine import AnalysisEngine, AnalysisEngineError
from modules.analysis.interpretation import explain_result
from modules.skills.data_analyst import build_analysis_plan


class DataAnalystWorkflow:
    """Run the non-UI part of the first Data Analyst closed loop."""

    def __init__(self, engine: AnalysisEngine | None = None):
        self.engine = engine or AnalysisEngine()

    def inspect_and_plan(
        self,
        file_path: str,
        requirement: str,
    ) -> dict[str, Any]:
        prepared = prepare_for_analysis(file_path)
        if not prepared.report.ready_for_analysis:
            return {
                "status": "blocked",
                "message": "数据质量检查未通过，请先处理阻塞问题。",
                "report": prepared.report,
            }

        plan = build_analysis_plan(
            requirement,
            prepared.report.columns,
            prepared.report.column_types,
        )
        if not plan["can_execute"]:
            return {
                "status": "need_clarification",
                "message": plan["question"],
                "report": prepared.report,
                "plan": plan,
            }

        return {
            "status": "ready",
            "message": "分析参数已齐全，请确认后执行。",
            "report": prepared.report,
            "plan": plan,
        }

    def execute(
        self,
        file_path: str,
        requirement: str,
    ) -> dict[str, Any]:
        prepared = prepare_for_analysis(file_path)
        if not prepared.report.ready_for_analysis:
            return {
                "status": "blocked",
                "message": "数据质量检查未通过，请先处理阻塞问题。",
                "report": prepared.report,
            }

        plan = build_analysis_plan(
            requirement,
            prepared.report.columns,
            prepared.report.column_types,
        )
        if not plan["can_execute"]:
            return {
                "status": "need_clarification",
                "message": plan["question"],
                "report": prepared.report,
                "plan": plan,
            }

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
        }
