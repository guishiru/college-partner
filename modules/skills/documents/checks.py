"""文档的确定性校验层：只查能机械判定的东西。

写 PRD 和写汇报稿，价值大半在追问和判断，那部分是大模型的事。但有一批问题
不需要判断，只需要数一数、算一算：必备章节缺没缺、标了 `[假设]` 有没有收进
索引、表格的合计行对不对、声称穷尽的百分比加不加得到 100。

这些放在代码里有两个好处：一是稳定，不会这次查出下次漏掉；二是可测——每条
规则都能用一份好稿和一份坏稿钉死。

**边界**：本模块只报「对不上」，不报「写得好不好」。是否 MECE、结论对不对、
排序合不合理，都需要理解内容，留给大模型和人。越过这条线，校验器就会开始
产生它自己也说不清的误报。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# 三级标注。文字标签不用 emoji：文档要往各种平台贴，符号会丢。
TAG_ASSUMPTION = "[假设]"
TAG_PENDING = "[待确认]"
TAG_TODO = "[待补充]"
TAGS = (TAG_ASSUMPTION, TAG_PENDING, TAG_TODO)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
# 形如 “1234 + 5678 = 6912” 或 “164362 / 8905 = 18.5”
_EQUATION = re.compile(
    r"(?P<left>-?[\d,]+(?:\.\d+)?(?:\s*[+\-*/×÷]\s*-?[\d,]+(?:\.\d+)?)+)"
    r"\s*[=＝]\s*(?P<right>-?[\d,]+(?:\.\d+)?)\s*(?P<unit>%|％)?"
)
_TOTAL_LABEL = re.compile(r"^(合计|总计|小计|共计|总和|Total)\s*$", re.I)


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    location: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentReview:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ready_to_deliver(self) -> bool:
        return not any(f.severity == SEVERITY_ERROR for f in self.findings)

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)


# ---------------------------------------------------------------------------
# 结构
# ---------------------------------------------------------------------------


def headings(text: str) -> list[tuple[int, str]]:
    """返回 (层级, 标题文字)。"""

    return [(len(m.group(1)), m.group(2).strip()) for m in _HEADING.finditer(text)]


def check_required_sections(
    text: str, required: list[str], *, code: str = "missing_section"
) -> list[Finding]:
    """必备章节缺没缺。按标题文字包含匹配，容忍编号和后缀。"""

    present = [title for _, title in headings(text)]
    missing = [
        name for name in required
        if not any(name in title for title in present)
    ]
    if not missing:
        return []
    return [
        Finding(
            code,
            SEVERITY_ERROR,
            f"缺少必备章节：{'、'.join(missing)}。",
            details={"missing": missing, "present": present},
        )
    ]


def check_heading_depth(text: str, *, max_depth: int = 3) -> list[Finding]:
    """层级超载：超过三层，读者记不住自己在哪儿。"""

    deep = [title for level, title in headings(text) if level > max_depth]
    if not deep:
        return []
    return [
        Finding(
            "heading_too_deep",
            SEVERITY_WARNING,
            f"标题层级超过 {max_depth} 层：{'、'.join(deep[:5])}。"
            f"层级过深时读者会丢失位置感，考虑合并或拆成并列章节。",
            details={"headings": deep},
        )
    ]


def extract_heading_outline(text: str) -> str:
    """标题抽取测试：把各级标题连成一段。

    这一步是确定性的——抽取本身不需要判断。抽出来读得通不通，是人和大模型的事。
    """

    return "\n".join("  " * (level - 1) + title for level, title in headings(text))


# ---------------------------------------------------------------------------
# 三级标注
# ---------------------------------------------------------------------------


def collect_tags(text: str) -> dict[str, list[str]]:
    """把正文里的标注按类型收集起来，附带所在行。"""

    found: dict[str, list[str]] = {tag: [] for tag in TAGS}
    for line in text.splitlines():
        for tag in TAGS:
            if tag in line:
                found[tag].append(line.strip())
    return found


def check_assumptions_indexed(
    text: str, *, index_section: str = "假设索引"
) -> list[Finding]:
    """标了 `[假设]` 就必须进假设索引——标了不收等于没标。"""

    tags = collect_tags(text)
    assumptions = tags[TAG_ASSUMPTION]
    if not assumptions:
        return []

    index_body = _section_body(text, index_section)
    if index_body is None:
        return [
            Finding(
                "assumption_index_missing",
                SEVERITY_ERROR,
                f"正文里有 {len(assumptions)} 处 {TAG_ASSUMPTION} 标注，"
                f"但没有「{index_section}」章节。标了不收等于没标。",
                details={"assumptions": assumptions},
            )
        ]

    # 索引里的行数至少要覆盖正文的假设条数（表头和分隔行不算）
    rows = [
        line for line in index_body.splitlines()
        if line.strip().startswith("|") and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", line)
    ]
    body_rows = max(len(rows) - 1, 0)
    if body_rows < len(assumptions):
        return [
            Finding(
                "assumption_index_incomplete",
                SEVERITY_ERROR,
                f"正文有 {len(assumptions)} 处 {TAG_ASSUMPTION}，"
                f"但假设索引只有 {body_rows} 行。每个假设在被确认前都是风险，必须逐条收录。",
                details={"assumptions": assumptions, "index_rows": body_rows},
            )
        ]
    return []


def _section_body(text: str, name: str) -> str | None:
    """取出某个章节的正文（到下一个同级或更高级标题为止）。"""

    matches = list(_HEADING.finditer(text))
    for index, m in enumerate(matches):
        if name not in m.group(2):
            continue
        level = len(m.group(1))
        start = m.end()
        end = len(text)
        for later in matches[index + 1:]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        return text[start:end]
    return None


# ---------------------------------------------------------------------------
# 数据自洽
# ---------------------------------------------------------------------------
#
# 只报「对不上」，不报「数字本身对不对」。外部事实无法验证——不知道「7 月质检单
# 164362 个」是不是真的；但能验证这个数在文中前后一不一致、写出来的算式算不算
# 得出写下的结果。守住这条边界，误报才不会淹没真问题。


def _to_number(raw: str) -> float:
    return float(raw.replace(",", "").strip())


def check_equations(text: str, *, tolerance: float = 0.005) -> list[Finding]:
    """凡是写出了算式的，代入算一遍。"""

    findings = []
    checked = 0
    for m in _EQUATION.finditer(text):
        expression = m.group("left").replace(",", "").replace("×", "*").replace("÷", "/")
        if not re.fullmatch(r"[\d\.\s+\-*/]+", expression):
            continue
        try:
            actual = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 受限于上面的白名单
        except (SyntaxError, ZeroDivisionError, TypeError):
            continue
        claimed = _to_number(m.group("right"))
        if m.group("unit"):
            actual *= 100
        checked += 1
        if abs(actual - claimed) > max(abs(claimed) * tolerance, 1e-9):
            findings.append(
                Finding(
                    "equation_mismatch",
                    SEVERITY_ERROR,
                    f"算式对不上：{m.group(0).strip()}，代入计算应为 {actual:.4g}。",
                    location=m.group(0).strip(),
                    details={"claimed": claimed, "actual": actual},
                )
            )
    if checked and not findings:
        findings.append(
            Finding(
                "equations_consistent",
                SEVERITY_INFO,
                f"已验算 {checked} 处算式，全部自洽。",
                details={"checked": checked},
            )
        )
    return findings


def markdown_tables(text: str) -> list[list[list[str]]]:
    """把 markdown 表格解析成 行 -> 单元格。"""

    tables, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            current.append([c.strip() for c in stripped.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def check_table_totals(text: str, *, tolerance: float = 0.005) -> list[Finding]:
    """表格里写了合计行的，核对各分项之和。"""

    findings = []
    for table in markdown_tables(text):
        if len(table) < 3:
            continue
        total_rows = [r for r in table if r and _TOTAL_LABEL.match(r[0])]
        if not total_rows:
            continue
        detail_rows = [
            r for r in table[1:]
            if r and not _TOTAL_LABEL.match(r[0]) and r is not table[0]
        ]
        for total_row in total_rows:
            for column in range(1, len(total_row)):
                claimed_raw = _NUMBER.search(total_row[column] or "")
                if not claimed_raw:
                    continue
                parts = [
                    _to_number(_NUMBER.search(r[column]).group(0))
                    for r in detail_rows
                    if column < len(r) and _NUMBER.search(r[column] or "")
                ]
                if len(parts) < 2:
                    continue
                claimed = _to_number(claimed_raw.group(0))
                actual = sum(parts)
                if abs(actual - claimed) > max(abs(claimed) * tolerance, 1e-9):
                    findings.append(
                        Finding(
                            "table_total_mismatch",
                            SEVERITY_ERROR,
                            f"表格「{table[0][column] if column < len(table[0]) else column}」列的"
                            f"合计写的是 {claimed:g}，各分项之和为 {actual:g}。",
                            location=" | ".join(total_row),
                            details={"claimed": claimed, "actual": actual},
                        )
                    )
    return findings


def check_percentage_totals(text: str, *, tolerance: float = 0.6) -> list[Finding]:
    """声称穷尽的百分比，是否加得到 100%。"""

    findings = []
    for table in markdown_tables(text):
        if len(table) < 3:
            continue
        for column in range(len(table[0])):
            values = []
            for row in table[1:]:
                if column >= len(row) or _TOTAL_LABEL.match(row[0] or ""):
                    continue
                m = re.fullmatch(r"\s*(-?[\d,]+(?:\.\d+)?)\s*[%％]\s*", row[column] or "")
                if m:
                    values.append(_to_number(m.group(1)))
            if len(values) < 3:
                continue
            total = sum(values)
            if abs(total - 100) > tolerance:
                findings.append(
                    Finding(
                        "percentage_total_mismatch",
                        SEVERITY_WARNING,
                        f"表格「{table[0][column]}」列的百分比加总为 {total:.2f}%，不是 100%。"
                        f"若这一列本就不穷尽，请在表下注明，避免读者误读。",
                        location=table[0][column],
                        details={"total": total, "values": values},
                    )
                )
    return findings


def check_data_consistency(text: str) -> list[Finding]:
    """算术自洽的三项合一：算式、表格合计、百分比加总。

    同一列若已经被合计行核对抓到，就不再由百分比加总重复报一次。校验器一旦
    开始重复报同一件事，人就会开始整体忽略它。
    """

    equations = check_equations(text)
    totals = check_table_totals(text)
    percentages = [
        f for f in check_percentage_totals(text)
        if f.details.get("total") is None
        or not any(
            abs(f.details["total"] - (other.details.get("actual") or 0)) < 1e-6
            for other in totals
        )
    ]
    return equations + totals + percentages
