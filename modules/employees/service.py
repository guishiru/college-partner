"""Virtual employees: identity, allowed Skills, collaborators and workspace layout.

This module owns what a virtual employee *is*. It does not run any Skill, does
not touch user data and does not decide which employee a user picked.

## Why layout lives here

The workspace layout is an attribute of the employee, not a global setting for
the product. A data analyst produces wide statistical tables that the user
reads while typing the next question, so conversation and results belong
side by side. Writing a PRD is a document being drafted, which wants a
different split. A workplace coach mostly answers in prose, where a second
panel is empty furniture.

Deciding this once, globally, is how a product ends up with an empty right-hand
panel on three quarters of its screens. Putting ``layout`` on the employee means
the answer is given per employee, at the point where the difference is known,
and the front end renders whatever the employee declares.

## v0.1 scope

The registry is declared in code. The product charter puts employee
configuration under the administrator role, so this will eventually be stored
and editable; until there is a second working employee, a table in the database
would be ceremony around a constant. What matters now is that the rest of the
system reads employees through this interface rather than hard-coding them —
moving the source of truth later then touches only this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Workspace layouts. The front end renders one of these; nothing else in the
# backend interprets them.
LAYOUT_ANALYSIS_SPLIT = "analysis_split"      # 对话 + 结果工作区（宽表、质量状态常驻）
LAYOUT_DOCUMENT_SPLIT = "document_split"      # 对话 + 正在起草的文档
LAYOUT_CONVERSATION = "conversation"          # 纯对话，没有第二栏

LAYOUTS = {LAYOUT_ANALYSIS_SPLIT, LAYOUT_DOCUMENT_SPLIT, LAYOUT_CONVERSATION}


class EmployeeError(ValueError):
    """Raised when an employee is unknown or not open for use."""


@dataclass(frozen=True)
class Employee:
    employee_id: str
    name: str
    badge: str
    description: str
    layout: str
    skills: tuple[str, ...] = ()
    can_invoke: tuple[str, ...] = ()
    available: bool = False

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "badge": self.badge,
            "description": self.description,
            "layout": self.layout,
            "skills": list(self.skills),
            "can_invoke": list(self.can_invoke),
            "available": self.available,
        }


EMPLOYEES: tuple[Employee, ...] = (
    Employee(
        employee_id="data_analyst",
        name="数据分析师",
        badge="数",
        description="问卷统计 · SPSS 口径",
        # 结果是宽表，而且用户要一边看结果一边提下一个需求。
        layout=LAYOUT_ANALYSIS_SPLIT,
        skills=("data_analyst.inspect", "data_analyst.analyze"),
        can_invoke=("report_writer",),   # 分析结果交给汇报文档写成结论先行的汇报
        available=True,
    ),
    Employee(
        employee_id="prd_writer",
        name="PRD撰写",
        badge="P",
        description="把模糊需求转成研发能接的 PRD",
        # 产出是一份正在成形的文档，右侧放文档本身。
        layout=LAYOUT_DOCUMENT_SPLIT,
        skills=("prd_writer.draft", "prd_writer.review"),
        can_invoke=(),
        available=True,
    ),
    Employee(
        employee_id="report_writer",
        name="汇报文档",
        badge="汇",
        # 与 PRD 撰写的分工：PRD 写给研发看，要的是完备；汇报写给要做决定的人
        # 看，要的是取舍——读者只看第一段就得拿到结论和要他做的事。
        description="结论先行的汇报稿，写 / 审 / 改",
        layout=LAYOUT_DOCUMENT_SPLIT,
        skills=("report_writer.draft", "report_writer.audit", "report_writer.revise"),
        can_invoke=(),
        available=True,
    ),
    Employee(
        employee_id="custom_data",
        name="定制数据专家",
        badge="定",
        description="按方法生成与校验数据",
        layout=LAYOUT_ANALYSIS_SPLIT,
        skills=(),
        can_invoke=(),
        available=False,
    ),
    Employee(
        employee_id="career_coach",
        name="职场陪跑顾问",
        badge="职",
        description="简历 · 面试 · 汇报",
        # 回答基本是成段文字，第二栏会一直空着。
        layout=LAYOUT_CONVERSATION,
        skills=(),
        can_invoke=(),
        available=False,
    ),
)

_BY_ID = {employee.employee_id: employee for employee in EMPLOYEES}


def list_employees() -> list[Employee]:
    return list(EMPLOYEES)


def get_employee(employee_id: str) -> Employee:
    employee = _BY_ID.get(employee_id or "")
    if employee is None:
        raise EmployeeError(f"虚拟员工不存在：{employee_id}")
    return employee


def require_available(employee_id: str) -> Employee:
    """Resolve an employee the user is allowed to start working with."""

    employee = get_employee(employee_id)
    if not employee.available:
        raise EmployeeError(f"{employee.name}尚未开放。")
    return employee


def can_invoke(caller_id: str, target_id: str) -> bool:
    """Whether one employee is configured to summon another."""

    return target_id in get_employee(caller_id).can_invoke
