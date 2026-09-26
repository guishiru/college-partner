"""事实清单与数字防线。

两个 skill 都写着同一条硬规则：**事实、数字、时间、人名只能来自用户给的材料，
缺了就标注，不要编。** 「预计提升 30%」这种数字编一个比空着危险得多——它会被
当真写进考核。

规则写在提示词里，模型多半会守，但「多半」不是工程。这里把它变成一道会拦截的
闸门：

1. 代码先从用户给的材料里抽出**事实清单**，每条带出处；
2. 模型只能用清单里的数；
3. 成稿回来后扫描其中每一个数字，不在清单里的，整稿打回重写。

## 哪些数字不算「编造」

全查会淹在误报里，所以放过这几类，它们不承载事实：

- 章节编号、条目编号（`1.`、`FR-003`、`4.2`）
- 小整数（默认 0-12）——「三个步骤」「两条路径」这类计数，以及月份
- 百分号不带数值的占位
- 标注为 `[待确认]`、`[待补充]` 的行

放过小整数是个取舍：它让「提升 3 倍」这种编造漏网。代价是确定的，收益是校验器
不会因为满屏误报而被人整体忽略——后者更致命。要更严就把 `small_number_limit`
调低。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .checks import SEVERITY_ERROR, TAGS, Finding

#: 正文里出现的数字。带千分位和小数。
_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
#: 章节号、条目号这类不承载事实的数字
_ORDINAL_CONTEXT = re.compile(r"(?:^|\s)(?:#{1,6}\s*)?\d+(?:\.\d+)*[、.）)]\s")
_ID_PATTERN = re.compile(r"[A-Za-z]+-\d+")

DEFAULT_SMALL_NUMBER_LIMIT = 12


@dataclass(frozen=True)
class Fact:
    """一条可引用的事实。``source`` 说明它从哪儿来。"""

    value: str
    label: str = ""
    source: str = ""

    def as_prompt_line(self) -> str:
        parts = [self.label, self.value]
        line = "：".join(p for p in parts if p)
        return f"- {line}（来源：{self.source}）" if self.source else f"- {line}"


@dataclass
class FactSheet:
    """给模型的事实清单。模型只能用这里面的数。"""

    facts: list[Fact] = field(default_factory=list)
    small_number_limit: int = DEFAULT_SMALL_NUMBER_LIMIT

    def add(self, value: Any, label: str = "", source: str = "") -> None:
        self.facts.append(Fact(value=_canonical(value), label=label, source=source))

    def allowed_numbers(self) -> set[str]:
        found = set()
        for fact in self.facts:
            for match in _NUMBER_IN_TEXT.finditer(fact.value):
                found.add(_canonical(match.group(0)))
        return found

    def as_prompt(self) -> str:
        if not self.facts:
            return "（没有可引用的事实。正文中不得出现任何具体数字。）"
        return "\n".join(fact.as_prompt_line() for fact in self.facts)

    def check(self, text: str) -> list[Finding]:
        """扫描成稿，找出清单之外的数字。"""

        allowed = self.allowed_numbers()
        invented: list[str] = []
        for line in _material_lines(text):
            for match in _NUMBER_IN_TEXT.finditer(line):
                raw = match.group(0)
                if _is_exempt(raw, line, match.start(), self.small_number_limit):
                    continue
                if _canonical(raw) not in allowed:
                    invented.append(raw)

        if not invented:
            return []
        unique = sorted(set(invented), key=invented.index)
        return [
            Finding(
                "number_not_in_fact_sheet",
                SEVERITY_ERROR,
                f"正文出现了事实清单里没有的数字：{'、'.join(unique[:8])}。"
                f"数字只能来自用户给的材料，缺了就标注 {TAGS[1]}，不要编——"
                f"编一个看起来合理的数比空着危险得多。",
                details={"numbers": unique, "allowed": sorted(allowed)},
            )
        ]


def _material_lines(text: str) -> list[str]:
    """跳过已标注为待补的行——那些行本来就是在说「这里还没有数」。"""

    return [
        line for line in (text or "").splitlines()
        if not any(tag in line for tag in TAGS) and "【待补" not in line
    ]


def _is_exempt(raw: str, line: str, position: int, small_limit: int) -> bool:
    prefix = line[:position]
    # FR-003 这类编号
    if _ID_PATTERN.search(line[max(0, position - 8):position + len(raw) + 2]):
        return True
    # 章节号 / 条目号
    if _ORDINAL_CONTEXT.match(prefix[-8:] + raw + (line[position + len(raw):position + len(raw) + 2])):
        return True
    if prefix.rstrip().endswith("#") or re.fullmatch(r"\s*#{1,6}\s*", prefix):
        return True
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return True
    # 小整数：计数、月份这类，不承载事实
    return value.is_integer() and 0 <= value <= small_limit


def _canonical(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def fact_sheet_from_analysis(results: list[dict[str, Any]]) -> FactSheet:
    """把分析结果转成事实清单。

    这是「报告只能引用已授权的分析结果」那条约定的落地形式：汇报文档拿到的
    不是一堆表，而是一份带出处的事实清单，写出来的每个数都能回溯到某个任务。
    """

    sheet = FactSheet()
    for result in results or []:
        provenance = result.get("provenance") or {}
        source = provenance.get("job_id") or result.get("method") or "分析结果"
        label_prefix = result.get("label") or result.get("method") or ""
        for table in result.get("tables") or []:
            frame = table.get("data")
            if frame is None:
                continue
            for column in getattr(frame, "columns", []):
                series = frame[column]
                if not hasattr(series, "dtype"):
                    continue
                for value in series.tolist():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if value != value:          # NaN
                        continue
                    sheet.add(value, label=f"{label_prefix}·{column}", source=source)
        meta = result.get("meta") or {}
        for key, value in meta.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sheet.add(value, label=f"{label_prefix}·{key}", source=source)
    return sheet
