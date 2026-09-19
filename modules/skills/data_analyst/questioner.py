"""Clarification questions for incomplete analysis plans."""


def next_question(missing: list[dict], columns: list[str]) -> str:
    if not missing:
        return ""
    item = missing[0]
    question = item["question"]
    candidates = "、".join(columns[:18])
    if candidates:
        return f"{question}\n当前可选列：{candidates}"
    return question
