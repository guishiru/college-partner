"""汇报文档 Skill：结论先行的结构模板与审阅校验。"""

from modules.skills.report_writer.service import (
    draft_document,
    draft_skeleton,
    DRAFT_SYSTEM,
    AUDIT_SECTIONS,
    EMPTY_HEADING_TERMS,
    REQUIRED_CONTEXT,
    SKELETON_CONFIRM_PROMPT,
    SKELETON_TEMPLATE,
    VERDICTS,
    audit_outline,
    review_report,
    verdict_is_consistent,
)

__all__ = [
    "draft_document",
    "draft_skeleton",
    "DRAFT_SYSTEM",
    "AUDIT_SECTIONS",
    "EMPTY_HEADING_TERMS",
    "REQUIRED_CONTEXT",
    "SKELETON_CONFIRM_PROMPT",
    "SKELETON_TEMPLATE",
    "VERDICTS",
    "audit_outline",
    "review_report",
    "verdict_is_consistent",
]
