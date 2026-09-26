"""汇报文档：结论先行的结构模板与审阅校验。

## 和 PRD 撰写的分工

PRD 写给研发看，要的是完备——每条需求都能拆成任务，每个异常都有交代，篇幅
长没关系。汇报写给要做决定的人看，要的是**取舍**——读者只看第一段就得拿到
结论和要他做的事，其余是支撑。

两者共用「先出骨架、停下来等确认」和「素材不够就标注不要编」，但质检标准
完全不同：PRD 查完备度，汇报查论证结构。

## 四条质检标准：论、证、类、比

| 字 | 标准 | 自检问法 |
|---|---|---|
| 论 | 结论先行 | 读者只看第一段，能不能拿到核心结论和要他做的事 |
| 证 | 以上统下 | 每个下级要点，是不是都在回答上级抛出的那个问题 |
| 类 | 归类分组 | 同层要点之间有没有重叠？加起来有没有遗漏 |
| 比 | 逻辑递进 | 同一组要点是按同一种顺序排的吗 |

其中「类」和「比」需要理解内容，代码判不了，留给大模型。本模块只做「论」的
形式检查和「证」的标题抽取，以及先于结构诊断的数据自洽——**算术错误比结构
问题更急**：读者验算一次发现错了，后面所有数字他都不会再信。
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
    TAG_TODO,
    DocumentReview,
    Finding,
    check_data_consistency,
    check_heading_depth,
    collect_tags,
    extract_heading_outline,
    headings,
)

# --- 流程层 -----------------------------------------------------------------

SKELETON_TEMPLATE = """【中心思想】一句话，必须是结论不是主题
【关键句】3-4 条，每条都是完整结论句
【序言 SCQA】情景 / 冲突 / 疑问 / 回答，各一到两句
【建议篇幅】约 N 字，M 个章节"""

SKELETON_CONFIRM_PROMPT = "骨架对吗？要调整哪里？确认后我展开成稿。"

#: 起草前必须问清的两件事，缺一不可。
REQUIRED_CONTEXT = (
    "读者是谁，以及看完之后你希望他做什么（具体行为，不是「让他了解」）",
    "读者已经知道哪些背景（决定 SCQA 里的 S 写什么）",
)

#: 审模式的输出格式，七段，顺序不变。
AUDIT_SECTIONS = (
    "场景判断", "总体判断", "做得好的", "数据自洽",
    "中心思想", "序言检查", "问题清单", "修改优先级",
)

VERDICTS = ("可直接发", "局部调整", "需动主干")

#: 反模式 7：无信息标题。这些词单独成标题时不承载任何结论。
EMPTY_HEADING_TERMS = (
    "现状分析", "现状", "背景介绍", "背景", "问题与挑战", "存在的问题",
    "下一步", "下一步计划", "工作总结", "总结", "概述", "相关情况",
    "基本情况", "其他", "附录说明", "思考与建议", "几点思考",
)

#: 反模式 2：结论被埋。开头是这些写法时，第一段没有结论。
NON_CONCLUSION_OPENINGS = (
    "为了", "根据", "按照", "近期", "近日", "本月", "上月", "本周",
    "自", "随着", "在", "为贯彻", "为落实",
)


# --- 确定性层 ---------------------------------------------------------------


def review_report(text: str) -> DocumentReview:
    """按审模式把一份汇报稿过一遍确定性检查。

    数据自洽先于结构诊断——顺序不是排版，是优先级。
    """

    review = DocumentReview()
    review.extend(check_data_consistency(text))
    review.extend(_check_conclusion_first(text))
    review.extend(_check_informative_headings(text))
    review.extend(check_heading_depth(text))
    review.extend(_check_unfilled_placeholders(text))
    return review


def _first_paragraph(text: str) -> str:
    """跳过标题，取正文第一段。"""

    body = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.M)
    for chunk in body.split("\n\n"):
        cleaned = chunk.strip()
        if cleaned and not cleaned.startswith("|"):
            return cleaned
    return ""


def _check_conclusion_first(text: str) -> list[Finding]:
    """「论」：读者只看第一段，能不能拿到结论和要他做的事。

    代码判不了「这句是不是结论」，但判得了两个形式信号：第一段是不是以
    铺垫句式开头，以及全文有没有出现过要读者做什么的句子。
    """

    findings: list[Finding] = []
    opening = _first_paragraph(text)
    if not opening:
        return [
            Finding("no_body_text", SEVERITY_ERROR, "文档没有正文段落。")
        ]

    head = opening[:24]
    hit = [w for w in NON_CONCLUSION_OPENINGS if head.startswith(w)]
    if hit:
        findings.append(
            Finding(
                "conclusion_buried",
                SEVERITY_WARNING,
                f"第一段以「{hit[0]}」开头，是铺垫不是结论。"
                f"汇报的读者只看第一段，把结论和要他做的事提到最前面。",
                location=opening[:60],
            )
        )

    if not re.search(r"(建议|申请|需要您|请批|请确认|拟|决策点|需决策)", text):
        findings.append(
            Finding(
                "no_ask",
                SEVERITY_WARNING,
                "全文没有出现要读者做什么。汇报的目标不是「让他了解」，"
                "而是具体行为——批预算、拍板方案、调资源。没有诉求就不必发。",
            )
        )
    return findings


def _check_informative_headings(text: str) -> list[Finding]:
    """反模式 7：无信息标题。标题抽出来连读要能拿到完整论证。"""

    titles = [title for level, title in headings(text) if level > 1]
    if not titles:
        return []
    empty = [t for t in titles if t.strip() in EMPTY_HEADING_TERMS]
    if not empty:
        return [
            Finding(
                "headings_informative",
                SEVERITY_INFO,
                "各级标题均承载信息。建议把标题单独抽出来连读，确认论证完整。",
                details={"outline": extract_heading_outline(text)},
            )
        ]
    return [
        Finding(
            "heading_without_information",
            SEVERITY_WARNING,
            f"这些标题只说了类别，没说结论：{'、'.join(empty)}。"
            f"把标题单独抽出来连读，应该能拿到完整论证——"
            f"「现状分析」读完不知道现状如何，改成结论句。",
            details={"headings": empty, "outline": extract_heading_outline(text)},
        )
    ]


def _check_unfilled_placeholders(text: str) -> list[Finding]:
    """素材不够就标注，但标注不能带着发出去。"""

    placeholders = collect_tags(text)[TAG_TODO]
    inline = re.findall(r"【待补[：:][^】]*】", text)
    total = len(placeholders) + len(inline)
    if not total:
        return []
    return [
        Finding(
            "placeholder_not_filled",
            SEVERITY_ERROR,
            f"文中还有 {total} 处待补标注，不能直接发出。"
            f"缺的素材要么去要，要么删掉该段——不要编一个看起来合理的数。",
            details={"placeholders": placeholders + inline},
        )
    ]


def audit_outline(text: str) -> str:
    """标题抽取测试的原料：把各级标题连成一段，交给人或大模型判断。"""

    return extract_heading_outline(text)


def verdict_is_consistent(verdict: str, findings: list[Finding]) -> bool:
    """总体判断必须和问题清单一致。

    清单里有动主干的问题，就不能判「可直接发」——这条规则在规程里写了，
    但只有变成代码才不会在赶时间的时候被跳过。
    """

    if verdict not in VERDICTS:
        raise ValueError(f"总体判断只能是 {VERDICTS} 之一，收到：{verdict}")
    has_error = any(f.severity == SEVERITY_ERROR for f in findings)
    has_warning = any(f.severity == SEVERITY_WARNING for f in findings)
    if verdict == "可直接发":
        return not has_error and not has_warning
    if verdict == "局部调整":
        return not has_error
    return True


# --- 生成层 -----------------------------------------------------------------

DRAFT_SYSTEM = (
    "你是汇报文档撰写助手。读者是要做决定的人，不是要了解情况的人。\n\n"
    "四条质检标准：\n"
    "- 论：结论先行。读者只看第一段，就要拿到核心结论和要他做的事。\n"
    "- 证：以上统下。每个下级要点都在回答上级抛出的那个问题。\n"
    "- 类：归类分组。同层要点之间不重叠，加起来不遗漏。\n"
    "- 比：逻辑递进。同一组要点按同一种顺序排——时间、结构、重要性，三选一。\n\n"
    "硬规则：\n"
    "1. 第一段必须是结论加诉求，不要用「为了」「根据」「近期」这类铺垫开头。\n"
    "2. 标题要承载结论。「现状分析」这种只说类别的标题不合格。\n"
    "3. 标题层级不超过三层。\n"
    "4. 事实、数字、时间、人名只能来自给定材料。缺了就写【待补：具体缺什么】，不要编。\n"
    "5. 写出算式的地方要算得对；表格有合计行的，分项之和要等于合计。\n\n"
    "只输出 Markdown 正文，不要解释。"
)

SKELETON_SYSTEM = (
    "你是汇报文档撰写助手。第一轮只出骨架，不要写成稿。严格按这个格式输出：\n\n"
    + SKELETON_TEMPLATE + "\n\n不要输出其他内容。"
)


def draft_skeleton(brief: str, *, client=None, fact_sheet: FactSheet | None = None) -> DraftResult:
    """第一轮：只出骨架，等用户确认。"""

    return generate_skeleton(
        client=client, system=SKELETON_SYSTEM, brief=brief, fact_sheet=fact_sheet
    )


def draft_document(brief: str, *, client=None, fact_sheet: FactSheet | None = None,
                   max_rewrites: int | None = None) -> DraftResult:
    """用户确认骨架后成稿，并过审模式校验回路。"""

    kwargs = {} if max_rewrites is None else {"max_rewrites": max_rewrites}
    return generate_with_review(
        client=client, system=DRAFT_SYSTEM, brief=brief,
        review_fn=review_report, fact_sheet=fact_sheet, **kwargs
    )
