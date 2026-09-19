"""Deterministic first-pass explanations for structured analysis results."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def explain_result(result: dict[str, Any]) -> str:
    method = result.get("method")
    label = result.get("label", method)
    meta = result.get("meta") or {}
    tables = result.get("tables") or []
    sample_size = meta.get("analysis_sample_size", meta.get("n", "未记录"))

    if method == "reliability":
        alpha = _find_value(tables, "Cronbach's α系数")
        if alpha is not None:
            return (
                f"本次{label}使用 {sample_size} 个有效样本。"
                f"Cronbach α 为 {alpha:.3f}，请结合量表所属维度和题项统计判断内部一致性。"
            )
    if method == "validity":
        kmo = _find_value(tables, "子项", "KMO值")
        p_value = _find_value(tables, "子项", "P")
        if kmo is not None:
            p_text = f"，Bartlett 检验结果为 {p_value}" if p_value is not None else ""
            return (
                f"本次{label}使用 {sample_size} 个有效样本。"
                f"KMO 值为 {kmo:.3f}{p_text}，用于判断数据是否适合继续进行因子分析。"
            )
    if method == "regression":
        dependent = meta.get("dependent", "未记录")
        independents = "、".join(meta.get("independents") or [])
        return (
            f"本次{label}使用 {sample_size} 个有效样本，"
            f"因变量为 {dependent}，自变量为 {independents}。"
            "请结合模型汇总、回归系数、显著性和共线性指标解读。"
        )
    if method == "correlation":
        return (
            f"本次{label}使用 {sample_size} 个有效样本，"
            "已生成 Pearson 相关矩阵。相关关系不等同于因果关系。"
        )
    if method == "frequency":
        return f"本次{label}使用 {sample_size} 个有效样本，已生成各变量的频数和百分比分布。"
    if method == "descriptive":
        return f"本次{label}使用 {sample_size} 个有效样本，已生成集中趋势和离散趋势统计量。"
    return (
        f"本次{label}已完成，使用 {sample_size} 个有效样本。"
        "请结合结果表中的估计值、显著性和模型前提进行判断。"
    )


def _find_value(tables, *keys):
    for table in tables:
        frame = table.get("data")
        if not isinstance(frame, pd.DataFrame):
            continue
        if len(keys) == 1 and keys[0] in frame.columns and not frame.empty:
            value = frame.iloc[0][keys[0]]
            return _number(value)
        if len(keys) == 2 and all(key in frame.columns for key in keys):
            rows = frame[frame[keys[0]].astype(str) == keys[1]]
            if not rows.empty:
                return _number(rows.iloc[0].iloc[-1])
    return None


def _number(value):
    if isinstance(value, str):
        cleaned = value.replace("*", "").replace("<", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return None
