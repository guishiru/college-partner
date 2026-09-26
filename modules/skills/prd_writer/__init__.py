"""PRD 撰写 Skill：章节模板、完备度闸门与交付前校验。"""

from modules.skills.prd_writer.service import (
    draft_document,
    draft_skeleton,
    DRAFT_SYSTEM,
    COMPLETENESS_FIELDS,
    EXCEPTION_CATEGORIES,
    REQUIRED_SECTIONS,
    SKELETON_CONFIRM_PROMPT,
    SKELETON_TEMPLATE,
    UNOBSERVABLE_TERMS,
    completeness_gate,
    review_prd,
)

__all__ = [
    "draft_document",
    "draft_skeleton",
    "DRAFT_SYSTEM",
    "COMPLETENESS_FIELDS",
    "EXCEPTION_CATEGORIES",
    "REQUIRED_SECTIONS",
    "SKELETON_CONFIRM_PROMPT",
    "SKELETON_TEMPLATE",
    "UNOBSERVABLE_TERMS",
    "completeness_gate",
    "review_prd",
]
