"""Unified deterministic entry point for the 11 statistical methods.

The language layer will eventually produce the ``method`` and ``params``
payload consumed here. This module deliberately does not infer intent, clean
uploaded files, or call an LLM. It validates a prepared DataFrame, invokes one
fixed method implementation, and normalizes its structured result.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class AnalysisEngineError(RuntimeError):
    """Base error for deterministic analysis execution."""


class AnalysisInputError(AnalysisEngineError):
    """Raised when method parameters or data do not satisfy a method contract."""


class AnalysisDependencyError(AnalysisEngineError):
    """Raised when an optional method dependency is unavailable."""


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    module: str
    function: str
    family: str
    min_cols: int = 0


METHOD_SPECS: dict[str, MethodSpec] = {
    "descriptive": MethodSpec(
        "descriptive", "描述统计", "descriptive_analysis", "descriptive_analysis", "cols", 1
    ),
    "frequency": MethodSpec(
        "frequency", "频数分析", "frequency_analysis", "frequency_analysis", "any_cols", 1
    ),
    "reliability": MethodSpec(
        "reliability", "信度分析", "reliability_analysis", "reliability_analysis", "numeric_cols", 2
    ),
    "validity": MethodSpec(
        "validity", "效度分析", "validity_analysis", "validity_analysis", "numeric_cols", 2
    ),
    "correlation": MethodSpec(
        "correlation", "相关分析", "correlation_analysis", "correlation_analysis", "numeric_cols", 2
    ),
    "efa": MethodSpec(
        "efa", "探索性因子分析", "exploratory_factor_analysis", "exploratory_factor_analysis", "numeric_cols", 2
    ),
    "cfa": MethodSpec(
        "cfa", "验证性因子分析", "confirmatory_factor_analysis", "confirmatory_factor_analysis", "factors"
    ),
    "regression": MethodSpec(
        "regression", "线性回归", "linear_regression", "linear_regression", "regression"
    ),
    "moderation": MethodSpec(
        "moderation", "调节效应分析", "moderation_analysis", "moderation_analysis", "moderation"
    ),
    "parallel_mediation": MethodSpec(
        "parallel_mediation",
        "平行中介效应",
        "parallel_mediation_analysis",
        "parallel_mediation",
        "parallel_mediation",
    ),
    "sem": MethodSpec(
        "sem", "结构方程模型", "sem_analysis", "run_sem", "sem"
    ),
}


class AnalysisEngine:
    """Validate and execute the fixed statistical method implementations."""

    source_version = "v517-validation-baseline"

    def list_methods(self) -> list[dict[str, Any]]:
        return [
            {
                "key": spec.key,
                "label": spec.label,
                "family": spec.family,
                "min_cols": spec.min_cols,
            }
            for spec in METHOD_SPECS.values()
        ]

    def run(
        self,
        method: str,
        data: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = METHOD_SPECS.get(method)
        if spec is None:
            raise AnalysisInputError(f"不支持的统计方法：{method}")
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise AnalysisInputError("当前没有可分析的数据。")

        request = dict(params or {})
        normalized = self._validate_and_normalize(spec, data, request)
        function = self._load_function(spec)

        try:
            raw = self._invoke(spec, function, data.copy(), normalized)
        except AnalysisEngineError:
            raise
        except ImportError as exc:
            # Some methods import their heavy dependency at module level and
            # some inside the function. Either way a missing package is a
            # dependency problem, not a data problem, and callers need to be
            # able to tell the two apart.
            raise AnalysisDependencyError(
                f"{spec.label}依赖未安装或无法加载：{exc}"
            ) from exc
        except Exception as exc:
            raise AnalysisEngineError(f"{spec.label}执行失败：{exc}") from exc

        return self._normalize_result(spec, normalized, raw)

    def _load_function(self, spec: MethodSpec):
        try:
            module = importlib.import_module(
                f"{__package__}.methods.{spec.module}"
            )
        except ImportError as exc:
            raise AnalysisDependencyError(
                f"{spec.label}依赖未安装或无法加载：{exc}"
            ) from exc
        function = getattr(module, spec.function, None)
        if function is None:
            raise AnalysisEngineError(
                f"{spec.label}实现缺少入口函数：{spec.function}"
            )
        return function

    def _validate_and_normalize(
        self,
        spec: MethodSpec,
        data: pd.DataFrame,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if data.columns.duplicated().any():
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise AnalysisInputError(f"数据存在重复列名：{duplicates}")

        if spec.family in {"cols", "any_cols", "numeric_cols"}:
            cols = self._list_param(params, "cols", spec.label)
            if len(cols) < spec.min_cols:
                raise AnalysisInputError(
                    f"{spec.label}至少需要 {spec.min_cols} 个变量。"
                )
            self._validate_columns(data, cols)
            if spec.family == "numeric_cols":
                self._require_numeric(data, cols, spec.label)
                self._require_analyzable(data, cols, spec.label)
            else:
                # 描述统计和频数不需要变异（常数列的均值是有意义的），
                # 但全空列和成对剔除后样本归零同样必须拦住。
                self._require_analyzable(
                    data, cols, spec.label, needs_variance=False
                )
            return {"cols": cols, **self._optional(params, "n_factors")}

        if spec.family == "regression":
            y = self._single_param(params, "y", spec.label)
            x_cols = self._list_param(params, "x_cols", spec.label)
            if y in x_cols:
                raise AnalysisInputError("因变量不能同时作为自变量。")
            self._validate_columns(data, [y, *x_cols])
            self._require_numeric(data, [y, *x_cols], spec.label)
            self._require_analyzable(data, [y, *x_cols], spec.label)
            return {
                "y": y,
                "x_cols": x_cols,
                **self._optional(params, "save_path"),
            }

        if spec.family == "moderation":
            x = self._single_param(params, "x", spec.label)
            y = self._single_param(params, "y", spec.label)
            m = params.get("m")
            if isinstance(m, list):
                m_list = [str(item) for item in m if str(item).strip()]
            else:
                m_list = [str(m).strip()] if m else []
            if len(m_list) != 1:
                raise AnalysisInputError("调节效应分析需要 1 个调节变量 M。")
            controls = self._optional_list(params, "controls")
            cols = [x, y, m_list[0], *controls]
            self._validate_distinct_roles(cols, spec.label)
            self._validate_columns(data, cols)
            self._require_numeric(data, cols, spec.label)
            self._require_analyzable(data, cols, spec.label)
            return {"x": x, "y": y, "m": m_list[0], "controls": controls}

        if spec.family == "parallel_mediation":
            x = self._single_param(params, "x", spec.label)
            y = self._single_param(params, "y", spec.label)
            mediators = self._list_param(params, "m_cols", spec.label)
            controls = self._optional_list(params, "controls")
            cols = [x, *mediators, y, *controls]
            self._validate_distinct_roles(cols, spec.label)
            self._validate_columns(data, cols)
            self._require_numeric(data, cols, spec.label)
            self._require_analyzable(data, cols, spec.label)
            return {
                "x": x,
                "y": y,
                "m_cols": mediators,
                "controls": controls,
                "n_boot": int(params.get("n_boot", 1000)),
                "ci": float(params.get("ci", 0.95)),
                "seed": int(params.get("seed", 42)),
            }

        if spec.family in {"factors", "sem"}:
            factors = params.get("factors")
            if not isinstance(factors, dict) or not factors:
                raise AnalysisInputError(f"{spec.label}需要明确因子结构。")
            all_items: list[str] = []
            for name, items in factors.items():
                if not isinstance(name, str) or not name.strip():
                    raise AnalysisInputError(f"{spec.label}的因子名称不能为空。")
                if not isinstance(items, list) or len(items) < 2:
                    raise AnalysisInputError(f"{spec.label}中每个因子至少需要 2 个题项。")
                all_items.extend(str(item) for item in items)
            if len(set(all_items)) != len(all_items):
                raise AnalysisInputError(f"{spec.label}中题项不能重复归属于多个因子。")
            self._validate_columns(data, all_items)
            self._require_numeric(data, all_items, spec.label)
            self._require_analyzable(data, all_items, spec.label)
            if spec.family == "factors":
                return {
                    "factors": factors,
                    "factor_names": params.get("factor_names") or {},
                }
            paths = self._normalize_paths(params.get("paths"))
            if not paths:
                raise AnalysisInputError("结构方程模型至少需要 1 条路径。")
            factor_names = set(factors)
            if any(left not in factor_names or right not in factor_names for left, right in paths):
                raise AnalysisInputError("结构方程模型路径必须连接已定义的因子。")
            return {"factors": factors, "paths": paths}

        raise AnalysisInputError(f"未实现的参数校验类型：{spec.family}")

    def _invoke(self, spec: MethodSpec, function, data: pd.DataFrame, params: dict[str, Any]):
        if spec.family in {"cols", "any_cols", "numeric_cols"}:
            kwargs: dict[str, Any] = {"items": params["cols"]}
            if spec.key == "correlation":
                kwargs["method"] = "pearson"
            if "n_factors" in params and params["n_factors"] is not None:
                kwargs["n_factors"] = int(params["n_factors"])
            return function(data, **kwargs)
        if spec.family == "regression":
            return function(data, dependent=params["y"], independents=params["x_cols"])
        if spec.family == "moderation":
            return function(
                data,
                x=params["x"],
                y=params["y"],
                m=params["m"],
                controls=params["controls"] or None,
            )
        if spec.family == "parallel_mediation":
            return function(
                data,
                x=params["x"],
                mediators=params["m_cols"],
                y=params["y"],
                controls=params["controls"],
                n_boot=params["n_boot"],
                ci=params["ci"],
                seed=params["seed"],
            )
        if spec.family == "factors":
            return function(
                data,
                factors=params["factors"],
                factor_names=params["factor_names"],
            )
        if spec.family == "sem":
            return function(data, factors=params["factors"], paths=params["paths"])
        raise AnalysisEngineError(f"未实现的调用类型：{spec.family}")

    def _normalize_result(
        self,
        spec: MethodSpec,
        params: dict[str, Any],
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(raw, dict) or not isinstance(raw.get("tables"), dict):
            raise AnalysisEngineError(f"{spec.label}返回了不支持的结果结构。")

        display_tables = (raw.get("display") or {}).get("tables") or {}
        tables = []
        for key, value in raw["tables"].items():
            values = value if isinstance(value, list) else [value]
            for index, table in enumerate(values):
                if not isinstance(table, pd.DataFrame):
                    continue
                config = display_tables.get(key) or {}
                title = config.get("title") or key
                if len(values) > 1:
                    title = f"{title}（{index + 1}）"
                tables.append(
                    {
                        "key": key,
                        "title": title,
                        "data": table,
                        "display": config,
                    }
                )

        if not tables:
            raise AnalysisEngineError(f"{spec.label}没有返回可展示的结果表。")

        meta = dict(raw.get("meta") or {})
        meta.update(
            {
                "method_key": spec.key,
                "method_label": spec.label,
                "method_version": self.source_version,
                "params": params,
            }
        )
        return {
            "method": spec.key,
            "label": spec.label,
            "meta": meta,
            "tables": tables,
            "display": raw.get("display") or {},
        }

    @staticmethod
    def _list_param(params: dict[str, Any], key: str, label: str) -> list[str]:
        value = params.get(key)
        if not isinstance(value, list):
            raise AnalysisInputError(f"{label}的参数 {key} 必须是变量列表。")
        values = [str(item).strip() for item in value if str(item).strip()]
        if len(values) != len(set(values)):
            raise AnalysisInputError(f"{label}的参数 {key} 不能包含重复变量。")
        if not values:
            raise AnalysisInputError(f"{label}缺少参数：{key}。")
        return values

    @staticmethod
    def _single_param(params: dict[str, Any], key: str, label: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AnalysisInputError(f"{label}缺少参数：{key}。")
        return value.strip()

    @staticmethod
    def _optional(params: dict[str, Any], key: str) -> dict[str, Any]:
        return {key: params[key]} if key in params else {}

    @staticmethod
    def _optional_list(params: dict[str, Any], key: str) -> list[str]:
        value = params.get(key) or []
        if not isinstance(value, list):
            raise AnalysisInputError(f"参数 {key} 必须是列表。")
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _validate_columns(data: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise AnalysisInputError(f"数据中不存在这些变量：{missing}")

    @staticmethod
    def _require_numeric(data: pd.DataFrame, columns: list[str], label: str) -> None:
        non_numeric = [
            column
            for column in columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if non_numeric:
            raise AnalysisInputError(
                f"{label}要求变量为数值类型，当前不是数值类型：{non_numeric}。"
                "请先经过数据处理模块确认转换规则。"
            )
        if np.isinf(data[columns].to_numpy(dtype=float, copy=False)).any():
            raise AnalysisInputError(f"{label}的变量中存在无穷值。")

    @staticmethod
    def _require_analyzable(
        data: pd.DataFrame,
        columns: list[str],
        label: str,
        *,
        needs_variance: bool = True,
    ) -> None:
        """成对剔除之后还剩多少样本，够不够算出有意义的结果。

        这里拦的都是「能算出一个数、但那个数没有意义」的情况——它们不会抛异常，
        只会安静地返回垃圾：一个全空列会把同批变量的样本量一起清零，描述统计
        于是报告样本量 0；常数列参与信度会算出 -0.163 的负 α；n=2 的信度是
        完美的 1.0。这些都比报错危险，因为用户会照抄进论文。
        """

        if not columns:
            return

        empty = [c for c in columns if data[c].dropna().empty]
        if empty:
            raise AnalysisInputError(
                f"{label}无法进行：变量 {empty} 没有任何有效值。"
                f"分析采用成对剔除，纳入全空变量会把其余变量的样本量一并清零。"
                f"请先移除这些变量。"
            )

        effective = int(len(data[columns].dropna()))
        if effective == 0:
            raise AnalysisInputError(
                f"{label}无法进行：{columns} 没有一条同时在所有变量上都有值的记录，"
                f"成对剔除后有效样本量为 0。请检查缺失情况。"
            )
        if effective < 3:
            raise AnalysisInputError(
                f"{label}无法进行：有效样本量仅 {effective}，无法估计统计量。"
            )

        if not needs_variance:
            return

        constant = [c for c in columns if data[c].dropna().nunique() <= 1]
        if constant:
            raise AnalysisInputError(
                f"{label}无法进行：变量 {constant} 的所有取值相同（方差为 0），"
                f"无法计算相关、载荷或信度。请先移除这些变量。"
            )

        if len(columns) > 1 and effective <= len(columns):
            raise AnalysisInputError(
                f"{label}无法进行：有效样本量 {effective} 不大于变量个数 "
                f"{len(columns)}，协方差矩阵奇异，结果没有意义。"
            )

    @staticmethod
    def _validate_distinct_roles(columns: list[str], label: str) -> None:
        if len(columns) != len(set(columns)):
            raise AnalysisInputError(f"{label}的变量角色不能重复。")

    @staticmethod
    def _normalize_paths(value: Any) -> list[tuple[str, str]]:
        if not isinstance(value, list):
            return []
        paths = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                paths.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict) and item.get("from") and item.get("to"):
                paths.append((str(item["from"]), str(item["to"])))
        return paths
