"""Deterministic planning for the Data Analyst virtual employee."""

from __future__ import annotations

import re
from typing import Any

from .questioner import next_question
from .registry import METHOD_SKILLS, detect_methods


ROLE_LABELS = {
    "y": ["因变量", "被解释变量", "结果变量", "Y"],
    "x_cols": ["自变量", "解释变量", "预测变量", "X"],
    "x": ["自变量", "解释变量", "X"],
    "m": ["调节变量", "调节变量 M", "M"],
    "m_cols": ["中介变量", "中介", "M"],
    "controls": ["控制变量", "控制"],
}


def build_analysis_plan(
    text: str,
    columns: list[str],
    column_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Turn a user request into validated method/parameter candidates."""

    columns = [str(column) for column in columns]
    column_types = column_types or {}
    methods = detect_methods(text)
    if not methods:
        missing = [
            {
                "field": "method",
                "question": "请确认要做哪一种统计分析，例如信度、效度、相关、回归、中介、调节、EFA、CFA 或 SEM。",
            }
        ]
        return _plan([], missing, columns)

    items = []
    missing: list[dict[str, Any]] = []
    explicit_columns = match_columns(text, columns)
    for method in methods:
        spec = METHOD_SKILLS[method]
        params = _extract_params(method, text, columns, explicit_columns, column_types)
        items.append(
            {
                "method": method,
                "label": spec["label"],
                "params": params,
                "reason": f"识别到关键词：{_matched_keyword(text, spec['keywords'])}",
            }
        )
        missing.extend(_missing(method, params))

    return _plan(items, missing, columns)


def _plan(items, missing, columns):
    return {
        "methods": items,
        "missing": missing,
        "can_execute": bool(items) and not missing,
        "question": next_question(missing, columns),
    }


def _extract_params(method, text, columns, explicit_columns, column_types):
    kind = METHOD_SKILLS[method]["kind"]
    if kind == "cols":
        selected = explicit_columns[:]
        if not selected and _says_all(text):
            if METHOD_SKILLS[method]["numeric"]:
                selected = [
                    column for column in columns
                    if column_types.get(column) == "numeric"
                ]
            else:
                selected = columns[:]
        params = {"cols": selected}
        if method == "efa":
            factor_match = re.search(r"(\d+)\s*个?因子", text or "")
            if factor_match:
                params["n_factors"] = int(factor_match.group(1))
        return params
    if kind == "regression":
        return {
            "y": _first_role_column(text, columns, "y"),
            "x_cols": _role_columns(text, columns, "x_cols"),
        }
    if kind == "moderation":
        return {
            "x": _first_role_column(text, columns, "x"),
            "y": _first_role_column(text, columns, "y"),
            "m": _first_role_column(text, columns, "m"),
            "controls": _role_columns(text, columns, "controls"),
        }
    if kind == "parallel_mediation":
        return {
            "x": _first_role_column(text, columns, "x"),
            "y": _first_role_column(text, columns, "y"),
            "m_cols": _role_columns(text, columns, "m_cols"),
            "controls": _role_columns(text, columns, "controls"),
        }
    if kind in {"factors", "sem"}:
        factors = extract_factors(text, columns)
        params = {"factors": factors}
        if kind == "sem":
            params["paths"] = extract_paths(text, list(factors))
        return params
    return {}


def _missing(method, params):
    spec = METHOD_SKILLS[method]
    kind = spec["kind"]
    missing = []
    if kind == "cols":
        if len(params.get("cols") or []) < spec["min_cols"]:
            missing.append(
                {"method": method, "field": "cols", "question": spec["question"]}
            )
    elif kind == "regression":
        if not params.get("y"):
            missing.append({"method": method, "field": "y", "question": "线性回归需要明确因变量 Y。"})
        if not params.get("x_cols"):
            missing.append({"method": method, "field": "x_cols", "question": "线性回归需要至少 1 个自变量 X。"})
    elif kind == "moderation":
        for field, question in (
            ("x", "调节效应需要明确自变量 X。"),
            ("y", "调节效应需要明确因变量 Y。"),
            ("m", "调节效应需要明确调节变量 M。"),
        ):
            if not params.get(field):
                missing.append({"method": method, "field": field, "question": question})
    elif kind == "parallel_mediation":
        for field, question in (
            ("x", "中介效应需要明确自变量 X。"),
            ("y", "中介效应需要明确因变量 Y。"),
            ("m_cols", "中介效应需要至少 1 个中介变量。"),
        ):
            if not params.get(field):
                missing.append({"method": method, "field": field, "question": question})
    elif kind == "factors":
        if not _valid_factors(params.get("factors")):
            missing.append({"method": method, "field": "factors", "question": spec["question"]})
    elif kind == "sem":
        if not _valid_factors(params.get("factors")):
            missing.append({"method": method, "field": "factors", "question": "SEM 需要先明确因子结构。"})
        if not params.get("paths"):
            missing.append({"method": method, "field": "paths", "question": "SEM 需要至少 1 条路径关系。"})
    return missing


def match_columns(text: str, columns: list[str]) -> list[str]:
    """Match exact columns first, then dimension names such as A -> A1/A2."""

    text = text or ""
    matches = []
    for column in sorted(columns, key=len, reverse=True):
        if column and column in text and column not in matches:
            matches.append(column)
    for column in columns:
        base = re.sub(r"[_-]?\d+$", "", column)
        if base and base != column and base in text and column not in matches:
            matches.append(column)
    return matches


def extract_factors(text: str, columns: list[str]) -> dict[str, list[str]]:
    factors = {}
    for name, fragment in re.findall(
        r"([\u4e00-\u9fffA-Za-z0-9_]+)\s*[:：]\s*([^；;\n]+)",
        text or "",
    ):
        items = match_columns(fragment, columns)
        if len(items) >= 2:
            factors[name] = items
    return factors


def extract_paths(text: str, factor_names: list[str]) -> list[tuple[str, str]]:
    if not factor_names:
        return []
    escaped = "|".join(re.escape(name) for name in factor_names)
    paths = []
    for left, right in re.findall(
        rf"({escaped})\s*(?:->|→|影响|预测)\s*({escaped})",
        text or "",
    ):
        if left != right and (left, right) not in paths:
            paths.append((left, right))
    return paths


def _role_columns(text, columns, role):
    results = []
    labels = ROLE_LABELS[role]
    for label in labels:
        pattern = re.compile(
            rf"{re.escape(label)}\s*(?:为|是|=|:|：)?\s*([^；;\n。]+)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text or ""):
            fragment = match.group(1)
            matched = match_columns(fragment, columns)
            matched.sort(
                key=lambda column: (
                    fragment.find(column)
                    if column in fragment
                    else fragment.find(re.sub(r"[_-]?\d+$", "", column))
                )
            )
            for column in matched:
                if column not in results:
                    results.append(column)
    if role in {"x", "y", "m"}:
        return results[:1]
    return results


def _first_role_column(text, columns, role):
    values = _role_columns(text, columns, role)
    return values[0] if values else None


def _matched_keyword(text, keywords):
    lowered = (text or "").lower()
    return next((keyword for keyword in keywords if keyword.lower() in lowered), keywords[0])


def _valid_factors(factors):
    return isinstance(factors, dict) and bool(factors) and all(
        isinstance(name, str)
        and isinstance(items, list)
        and len(items) >= 2
        for name, items in factors.items()
    )


def _says_all(text):
    return bool(re.search(r"全部|所有|全量|所有题项|全部题项|全部变量|所有变量", text or ""))
