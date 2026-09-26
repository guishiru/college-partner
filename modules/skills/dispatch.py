"""按虚拟员工把写作请求分派到对应的 Skill。

放在 skills 包根而不是 documents 里：共用校验层在底层，两个员工在中间，
分派表在最上面。反过来会形成环——共用层去 import 员工，员工又 import 共用层。

Web 层不该知道「PRD撰写用哪个提示词、汇报文档用哪个校验器」——那是 Skill 层
的事。这里给它一张表：员工标识进来，骨架函数、成稿函数、审阅函数出去。

新增写作类员工时只动这张表，不动 web 层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from modules.skills.prd_writer import (
    draft_document as prd_draft,
    draft_skeleton as prd_skeleton,
    review_prd,
)
from modules.skills.report_writer import (
    audit_outline,
    draft_document as report_draft,
    draft_skeleton as report_skeleton,
    review_report,
)

from modules.skills.documents import DocumentReview


class UnsupportedEmployee(ValueError):
    """这个员工不做写作。"""


@dataclass(frozen=True)
class WritingSkill:
    employee_id: str
    skeleton: Callable
    draft: Callable
    review: Callable[[str], DocumentReview]
    skeleton_skill: str
    draft_skill: str
    review_skill: str


WRITING_SKILLS: dict[str, WritingSkill] = {
    "prd_writer": WritingSkill(
        employee_id="prd_writer",
        skeleton=prd_skeleton,
        draft=prd_draft,
        review=review_prd,
        skeleton_skill="prd_writer.skeleton",
        draft_skill="prd_writer.draft",
        review_skill="prd_writer.review",
    ),
    "report_writer": WritingSkill(
        employee_id="report_writer",
        skeleton=report_skeleton,
        draft=report_draft,
        review=review_report,
        skeleton_skill="report_writer.skeleton",
        draft_skill="report_writer.draft",
        review_skill="report_writer.audit",
    ),
}


def writing_skill_for(employee_id: str) -> WritingSkill:
    skill = WRITING_SKILLS.get(employee_id or "")
    if skill is None:
        raise UnsupportedEmployee(
            f"当前虚拟员工不提供文档撰写能力：{employee_id}。"
            f"请新建会话并选择 PRD撰写 或 汇报文档。"
        )
    return skill


def is_writing_employee(employee_id: str) -> bool:
    return employee_id in WRITING_SKILLS


__all__ = [
    "UnsupportedEmployee",
    "WRITING_SKILLS",
    "WritingSkill",
    "audit_outline",
    "is_writing_employee",
    "writing_skill_for",
]
