"""生成回路：出稿 → 校验 → 不过就带着问题清单重写。

## 为什么要有回路

校验器已经能指出「缺假设索引」「验收条件含流畅」「出现了清单外的数字」。把这
些问题原样递回给模型，比让人去改快得多，也比在提示词里反复叮嘱有效得多——
提示词是事前祈祷，校验是事后核对，后者才拦得住。

## 为什么重写要有上限

模型改不动的时候会原地打转：同一条问题改三次还在，再改三十次也一样，只是把钱
烧掉。到上限就停下来，把**当前稿加上未解决的问题清单**交还给用户——半成品加
一份明确的问题列表，比一份看起来完整但暗藏问题的稿子有用。

## 两阶段

产品约定「先出骨架，停下来等确认」，所以生成分两步，中间由用户拍板：

    skeleton()  只出骨架，不出成稿
    finalize()  用户确认骨架后才成稿，成稿过校验回路

例外：用户明确说「直接给我成稿」时，调用方可以跳过第一步。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from modules.llm import LLMClient, LLMError, LLMNotConfigured

from .checks import SEVERITY_ERROR, DocumentReview, Finding
from .facts import FactSheet

MAX_REWRITES = 2


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response) -> None:
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens


@dataclass
class DraftResult:
    """一次生成的产物，连同它没能解决的问题。"""

    text: str = ""
    review: DocumentReview = field(default_factory=DocumentReview)
    rewrites: int = 0
    usage: Usage = field(default_factory=Usage)
    failed: str = ""          # 非空表示没能出稿，内容是原因

    @property
    def ready_to_deliver(self) -> bool:
        return bool(self.text) and not self.failed and self.review.ready_to_deliver

    @property
    def unresolved(self) -> list[Finding]:
        return self.review.by_severity(SEVERITY_ERROR)


def _problem_list(review: DocumentReview) -> str:
    return "\n".join(
        f"{index}. [{finding.code}] {finding.message}"
        for index, finding in enumerate(review.by_severity(SEVERITY_ERROR), start=1)
    )


def generate_skeleton(
    *,
    client: LLMClient | None,
    system: str,
    brief: str,
    fact_sheet: FactSheet | None = None,
) -> DraftResult:
    """第一轮只出骨架。爆炸点几乎总在用户看到骨架的那一刻。"""

    result = DraftResult()
    if client is None:
        result.failed = "未配置模型"
        return result

    user = brief
    if fact_sheet is not None:
        user += "\n\n可引用的事实（只能用这些数字，缺的标注出来，不要编）：\n"
        user += fact_sheet.as_prompt()

    try:
        response = client.complete(system=system, user=user)
    except LLMNotConfigured:
        result.failed = "未配置模型"
        return result
    except LLMError as exc:
        result.failed = f"模型调用失败：{exc}"
        return result

    result.usage.add(response)
    result.text = response.text.strip()
    return result


def generate_with_review(
    *,
    client: LLMClient | None,
    system: str,
    brief: str,
    review_fn: Callable[[str], DocumentReview],
    fact_sheet: FactSheet | None = None,
    max_rewrites: int = MAX_REWRITES,
) -> DraftResult:
    """成稿并过校验回路。"""

    result = DraftResult()
    if client is None:
        result.failed = "未配置模型"
        return result

    user = brief
    if fact_sheet is not None:
        user += "\n\n可引用的事实（只能用这些数字，缺的标注出来，不要编）：\n"
        user += fact_sheet.as_prompt()

    for attempt in range(max_rewrites + 1):
        try:
            response = client.complete(system=system, user=user)
        except LLMNotConfigured:
            result.failed = "未配置模型"
            return result
        except LLMError as exc:
            # 已经出过稿就保留它，连同这次失败一起交还
            result.failed = f"模型调用失败：{exc}"
            return result

        result.usage.add(response)
        result.text = response.text.strip()
        result.rewrites = attempt

        review = review_fn(result.text)
        if fact_sheet is not None:
            review.extend(fact_sheet.check(result.text))
        result.review = review

        if review.ready_to_deliver:
            return result
        if attempt == max_rewrites:
            break

        user = (
            f"{brief}\n\n"
            f"你上一版的稿子没有通过交付前校验，问题如下，请逐条改掉后重出全文：\n"
            f"{_problem_list(review)}\n\n"
            f"只输出修改后的完整文稿，不要解释改了什么。\n\n"
            f"上一版：\n{result.text}"
        )

    return result
