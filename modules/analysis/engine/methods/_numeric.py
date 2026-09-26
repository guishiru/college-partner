"""数值列与展示渲染的统一约定。

## 约定

统计量一律以完整精度的数值输出，**不适用的位置存 NaN**，而不是塞一个 ``'-'``
字符串——一个 ``'-'`` 会把整列变成文本，连带把这一列其余的真数值也一起变成
字符串。渲染成什么样是展示层的事，写在 ``display`` 配置里。

显著性星号是标记不是数值，单独成列；p 值本身保持完整精度。前端要显示
``0.0008***`` 时，自己把两列拼起来。

## display 配置

每张表的展示配置可以带：

``numeric``
    ``{"na_text": "-", "decimals": {...}}``。``na_text`` 说明 NaN 渲染成什么；
    ``decimals`` 按列指定小数位，缺省由前端决定。
``combine``
    ``{"P值": "显著性"}``——渲染 ``P值`` 时把 ``显著性`` 列的星号接在后面。

后端不解释这些配置，它只是把渲染规则和数据放在一起，免得前端各自猜。

## 为什么不直接存格式化好的字符串

``'<0.001'`` 和 ``'0.0***'`` 不是数值而是一个界，报告模块无论怎么解析都拿不
回真值，只能自行重算——而项目约定禁止重算。详见
``docs/decisions/0004-统计量必须以数值形式输出.md``。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: 不适用的数值位置统一用它，而不是 '-'、'NaN'、'' 之类的占位字符串。
NA = np.nan

DEFAULT_NA_TEXT = "-"


def numeric_display(
    na_text: str = DEFAULT_NA_TEXT,
    decimals: dict[str, int] | None = None,
    combine: dict[str, str] | None = None,
) -> dict[str, Any]:
    """生成一张表的数值渲染配置，放进该表的 display 里。"""

    config: dict[str, Any] = {"numeric": {"na_text": na_text}}
    if decimals:
        config["numeric"]["decimals"] = dict(decimals)
    if combine:
        config["combine"] = dict(combine)
    return config


def to_numeric(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """把指定列转成数值，无法解析的位置变成 NaN。

    用于收尾：确保一列里混进来的占位符不会把整列拖成文本。
    """

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def as_float(value) -> float:
    """把可能是 None/空串/NaN 的值统一成 float，缺失即 NaN。"""

    if value is None or value == "":
        return NA
    try:
        result = float(value)
    except (TypeError, ValueError):
        return NA
    return result


# ---------------------------------------------------------------------------
# 显著性星号
# ---------------------------------------------------------------------------
#
# 口径在这里定义一次，所有方法共用。此前各方法各写一套，同一个 p=0.005 在相关
# 表里是 ***、在 SEM 表里是 **，两张表放进同一份报告就自相矛盾。
#
#   ***  p < 0.01
#   **   0.01 <= p < 0.05
#   *    0.05 <= p < 0.1
#   （无） p >= 0.1 或 p 缺失
#
# 改动口径请同步 docs/contracts/统计方法基线_v0.1.md 并重新生成黄金基线。

SIGNIFICANCE_THRESHOLDS = ((0.01, "***"), (0.05, "**"), (0.1, "*"))


def significance_star(p) -> str:
    """按统一口径返回显著性星号；p 缺失或无法解析时返回空串。"""

    value = as_float(p)
    if value != value:          # NaN
        return ""
    for threshold, star in SIGNIFICANCE_THRESHOLDS:
        if value < threshold:
            return star
    return ""
