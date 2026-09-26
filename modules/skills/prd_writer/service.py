"""PRD 撰写：章节模板、完备度闸门与交付前校验。

## 这个 Skill 的三层

**流程层**（本模块的常量）：骨架格式、提问优先级、必备章节、FR 写法。这些是
写作规程，由大模型执行。

**确定性层**（本模块的 `review_prd`）：能机械判定的部分——章节缺没缺、FR 三件
事齐不齐、验收条件里有没有不可观察的词、非功能需求有没有阀值、异常八类覆盖
了几类、标了 `[假设]` 有没有收进索引。

**判断层**：需求还原得对不对、范围切得合不合理、优先级排得对不对。这些是业务
决策，既不归代码也不归大模型——把笔还给用户。

## 为什么闸门要用代码实现

「出稿前逐项评一遍」写在规程里是靠自觉；同一份 PRD 今天查出五个问题、明天
查出三个，这样的闸门等于没有。放进代码，每次结果一样，而且能用一份好稿和一份
坏稿钉死。
"""

from __future__ import annotations

import re

from modules.skills.documents import (
    DraftResult,
    FactSheet,
    generate_skeleton,
    generate_with_review,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    TAG_PENDING,
    TAG_TODO,
    DocumentReview,
    Finding,
    check_assumptions_indexed,
    check_data_consistency,
    check_required_sections,
    collect_tags,
    headings,
)

# --- 流程层：写作规程 -------------------------------------------------------

#: 必备章节。按需拉取的（权限矩阵、状态机、接口约定、迁移方案等）不在此列。
REQUIRED_SECTIONS = [
    "修订记录", "背景与目标", "名词表", "范围", "用户与场景", "主流程",
    "功能需求", "异常与边界", "数据需求", "非功能需求", "依赖与风险",
    "发布策略", "成功指标", "待确认清单", "假设索引",
]

#: 第一轮只出骨架，不出成稿。爆炸点几乎总在「原来这个不做啊」。
SKELETON_TEMPLATE = """【需求一句话】为了让【角色】能【做到什么】，我们要【做什么】，预期带来【什么变化】
【做 / 不做】各 3-5 条
【功能点清单】FR-001… 每条一句话
【待确认清单】现在还不知道的，列出来
【预计篇幅】约 N 字，M 个章节"""

SKELETON_CONFIRM_PROMPT = "骨架对吗？特别看一下「不做」那几条。确认后我展开成稿。"

#: 完备度闸门。有红不准生成；有黄先告知用户哪几项单薄。
COMPLETENESS_FIELDS = {
    "用户与场景": ("说得出具体角色 + 使用频次", "只有角色没有场景", "只写「用户」"),
    "真正的需求": ("已从方案还原成需求并经用户确认", "还原了但没确认", "直接照抄业务方的方案"),
    "范围边界": ("做什么和明确不做什么都写了", "只写了做什么", "没边界"),
    "主流程": ("逐步可走通，分支有交代", "只有幸福路径", "没流程"),
    "验收标准": ("每条需求都有可验证的验收条件", "部分有", "没有"),
    "异常与边界": ("空/错/权限/重复 至少覆盖四类", "只写了报错", "没想"),
    "数据口径": ("每个业务名词有定义", "部分有", "全是模糊词"),
}

#: 异常与边界逐条过，没有就写「不适用」，不准留白。
EXCEPTION_CATEGORIES = {
    "空态": ("空态", "没数据", "无数据", "空列表", "空结果"),
    "错误态": ("错误态", "接口失败", "超时", "部分成功", "报错"),
    "加载态": ("加载态", "加载中", "等待", "loading"),
    "权限": ("权限", "无权", "越权", "可见性"),
    "重复与并发": ("重复", "并发", "连点", "同时"),
    "极端值": ("极端值", "超长", "负数", "上限", "0 条", "10 万"),
    "存量数据": ("存量", "历史数据", "老数据", "迁移"),
    "可逆性": ("可逆", "撤销", "回滚", "撤回"),
}

#: 验收条件里不可观察的说法。写「体验流畅」测试写不出用例。
UNOBSERVABLE_TERMS = (
    "流畅", "友好", "良好", "美观", "合理", "尽量", "尽快", "适当", "优化体验",
    "提升体验", "稳定可靠", "高效", "易用", "无感", "较快", "较好",
)

#: 非功能需求必须带阀值，否则是废话。
_THRESHOLD_PATTERN = re.compile(
    r"\d+\s*(ms|毫秒|s|秒|分钟|小时|天|%|％|条|次|万|k|K|M|G|QPS|TPS|P\d{2})"
)

_FR_HEADING = re.compile(r"(FR-\d+)")
_FR_PARTS = ("触发", "行为", "验收")


# --- 确定性层：交付前校验 ---------------------------------------------------


def review_prd(text: str) -> DocumentReview:
    """把一份 PRD 过一遍确定性闸门。"""

    review = DocumentReview()
    review.extend(check_required_sections(text, REQUIRED_SECTIONS))
    review.extend(_check_scope_has_exclusions(text))
    review.extend(_check_functional_requirements(text))
    review.extend(_check_exception_coverage(text))
    review.extend(_check_non_functional_thresholds(text))
    review.extend(check_assumptions_indexed(text))
    review.extend(_check_open_questions_collected(text))
    review.extend(check_data_consistency(text))
    return review


def _check_scope_has_exclusions(text: str) -> list[Finding]:
    """「明确不做」必须写。业务方对「做什么」通常没异议，爆炸点在「不做」。"""

    if re.search(r"(不做|非目标|本期不|暂不)", text):
        return []
    return [
        Finding(
            "scope_missing_exclusions",
            SEVERITY_ERROR,
            "范围里没有写「明确不做」。业务方对做什么通常没异议，"
            "返工几乎总是来自「原来这个不做啊」。写不出来就去问。",
        )
    ]


def _check_functional_requirements(text: str) -> list[Finding]:
    """每条 FR 固定四件：描述、触发、行为、验收；验收必须可观察。"""

    findings: list[Finding] = []
    blocks = _split_requirement_blocks(text)
    if not blocks:
        return [
            Finding(
                "no_functional_requirements",
                SEVERITY_ERROR,
                "没有找到任何 FR 编号的功能需求。功能需求必须逐条编号，研发才能拆任务。",
            )
        ]

    for name, body in blocks.items():
        missing = [part for part in _FR_PARTS if part not in body]
        if missing:
            findings.append(
                Finding(
                    "requirement_incomplete",
                    SEVERITY_ERROR,
                    f"{name} 缺少：{'、'.join(missing)}。"
                    f"每条功能需求固定四件：一句话描述、触发、行为、验收。",
                    location=name,
                    details={"missing": missing},
                )
            )
            continue

        acceptance = _part_body(body, "验收")
        vague = [term for term in UNOBSERVABLE_TERMS if term in acceptance]
        if vague:
            findings.append(
                Finding(
                    "acceptance_not_observable",
                    SEVERITY_ERROR,
                    f"{name} 的验收条件含不可观察的说法：{'、'.join(vague)}。"
                    f"测试写不出用例。改成「当…，若…，则…」这种可观察的结果。",
                    location=name,
                    details={"terms": vague, "acceptance": acceptance},
                )
            )
    return findings


def _split_requirement_blocks(text: str) -> dict[str, str]:
    """按 FR 编号切块。"""

    marks = list(_FR_HEADING.finditer(text))
    blocks: dict[str, str] = {}
    for index, m in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        blocks.setdefault(m.group(1), text[m.start():end])
    return blocks


def _part_body(block: str, label: str) -> str:
    m = re.search(rf"{label}\s*[:：]\s*(.*?)(?=\n\s*(?:触发|行为|验收)\s*[:：]|\n#|\Z)",
                  block, re.S)
    return m.group(1).strip() if m else ""


def _check_exception_coverage(text: str) -> list[Finding]:
    """八类异常逐条过，没有就写「不适用」，不准留白。"""

    from modules.skills.documents.checks import _section_body

    body = _section_body(text, "异常与边界")
    if body is None:
        return []          # 章节缺失已由必备章节检查报过，不重复
    missing = [
        name for name, keywords in EXCEPTION_CATEGORIES.items()
        if not any(keyword in body for keyword in keywords)
    ]
    if not missing:
        return [
            Finding(
                "exception_coverage_complete",
                SEVERITY_INFO,
                f"异常与边界八类均已覆盖。",
            )
        ]
    severity = SEVERITY_ERROR if len(missing) > 4 else SEVERITY_WARNING
    return [
        Finding(
            "exception_coverage_incomplete",
            severity,
            f"异常与边界未覆盖：{'、'.join(missing)}。"
            f"这些是研发最常回来追问的地方，不适用就写「不适用」，不要留白。",
            details={"missing": missing},
        )
    ]


def _check_non_functional_thresholds(text: str) -> list[Finding]:
    """没阀值的非功能需求是废话。"""

    from modules.skills.documents.checks import _section_body

    body = _section_body(text, "非功能需求")
    if body is None:
        return []
    lines = [
        line.strip(" -*·\t") for line in body.splitlines()
        if line.strip(" -*·\t") and not line.strip().startswith("|")
    ]
    offenders = [
        line for line in lines
        if not _THRESHOLD_PATTERN.search(line) and TAG_PENDING not in line
        and len(line) > 6
    ]
    if not offenders:
        return []
    return [
        Finding(
            "non_functional_without_threshold",
            SEVERITY_ERROR,
            f"非功能需求没有具体阀值：{'；'.join(offenders[:3])}。"
            f"「系统应具备良好的性能」这类写法删掉即可；拿不到阀值就标 {TAG_PENDING}，"
            f"不要填套话。",
            details={"lines": offenders},
        )
    ]


def _check_open_questions_collected(text: str) -> list[Finding]:
    """标了 [待确认]/[待补充] 就必须进待确认清单。"""

    from modules.skills.documents.checks import _section_body

    tags = collect_tags(text)
    pending = tags[TAG_PENDING] + tags[TAG_TODO]
    if not pending:
        return []
    body = _section_body(text, "待确认清单")
    if body is None or not body.strip():
        return [
            Finding(
                "open_questions_not_collected",
                SEVERITY_ERROR,
                f"正文有 {len(pending)} 处 {TAG_PENDING}/{TAG_TODO} 标注，"
                f"但待确认清单是空的。标了不收等于没标。",
                details={"pending": pending},
            )
        ]
    return []


def completeness_gate(fields: dict[str, str]) -> DocumentReview:
    """闸门评分。``fields`` 是每一项的判定：green / yellow / red。

    有红不准生成；有黄先告知用户哪几项单薄，问要不要补。
    """

    review = DocumentReview()
    unknown = sorted(set(fields) - set(COMPLETENESS_FIELDS))
    if unknown:
        raise ValueError(f"闸门中不存在这些字段：{unknown}")

    missing = sorted(set(COMPLETENESS_FIELDS) - set(fields))
    if missing:
        review.extend([
            Finding("gate_not_evaluated", SEVERITY_ERROR,
                    f"闸门未逐项评估：{'、'.join(missing)}。")
        ])
    reds = [k for k, v in fields.items() if v == "red"]
    yellows = [k for k, v in fields.items() if v == "yellow"]
    if reds:
        review.extend([
            Finding("gate_red", SEVERITY_ERROR,
                    f"以下项为红，不能出稿：{'、'.join(reds)}。",
                    details={"fields": reds})
        ])
    if yellows:
        review.extend([
            Finding("gate_yellow", SEVERITY_WARNING,
                    f"以下项单薄：{'、'.join(yellows)}。出稿前先问用户要不要补。",
                    details={"fields": yellows})
        ])
    return review


# --- 生成层 -----------------------------------------------------------------

DRAFT_SYSTEM = (
    "你是 PRD 撰写助手。把业务方的表述转成研发能直接接的 PRD。\n\n"
    "必备章节（顺序不变）：\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(REQUIRED_SECTIONS, 1)) + "\n\n"
    "硬规则：\n"
    "1. 「明确不做」必须写。业务方对做什么没异议，返工都来自「原来这个不做啊」。\n"
    "2. 每条功能需求用 FR-001 编号，固定四件：一句话描述、触发、行为、验收。\n"
    "3. 验收条件必须可观察。「体验流畅」不行，「点击后 1 秒内出现结果页」才行。\n"
    "4. 异常与边界逐条过这八类：" + "、".join(EXCEPTION_CATEGORIES) + "。没有就写「不适用」，不准留白。\n"
    "5. 非功能需求没有具体阀值就不写，或标 [待确认]。\n"
    "6. 你推断的内容标 [假设] 并逐条收进假设索引；不知道的标 [待确认] 并收进待确认清单。\n"
    "7. 时间、金额、量级、阀值、比例只能来自给定材料，不得编造。\n"
    "8. 不替用户做业务决策——做哪期、砍哪些功能、排什么优先级，是业务方的事。\n\n"
    "只输出 Markdown 正文，不要解释。"
)

SKELETON_SYSTEM = (
    "你是 PRD 撰写助手。第一轮只出骨架，不要写成稿。严格按这个格式输出：\n\n"
    + SKELETON_TEMPLATE + "\n\n不要输出其他内容。"
)


def draft_skeleton(brief: str, *, client=None, fact_sheet: FactSheet | None = None) -> DraftResult:
    """第一轮：只出骨架，等用户确认。"""

    return generate_skeleton(
        client=client, system=SKELETON_SYSTEM, brief=brief, fact_sheet=fact_sheet
    )


def draft_document(brief: str, *, client=None, fact_sheet: FactSheet | None = None,
                   max_rewrites: int | None = None) -> DraftResult:
    """用户确认骨架后成稿，并过交付前校验回路。"""

    kwargs = {} if max_rewrites is None else {"max_rewrites": max_rewrites}
    return generate_with_review(
        client=client, system=DRAFT_SYSTEM, brief=brief,
        review_fn=review_prd, fact_sheet=fact_sheet, **kwargs
    )
