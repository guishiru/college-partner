"""语义识别：关键词表兜不住的说法，交给大模型翻译。

## 为什么不是直接上模型

关键词表命中的时候，它比模型**又快又准又免费**，而且结果稳定。「做信度分析」
这种说法没有任何歧义，为它调一次模型是白花钱，还引入了一个本来不存在的
不确定性来源。所以顺序是：关键词先跑，兜不住才轮到模型。

实测兜不住的是这类：

    「看看社会支持这几道题一不一致」   → 关键词表识别不出
    「这三道题内部一致性怎么样」       → 关键词表识别不出
    「社会支持和主观规范有没有关系」   → 关键词表识别不出

这些都是人正常会说的话。

## 模型的笼子

模型只做翻译，不碰计算。它的输出在进入引擎之前必须过四道校验：

1. **方法必须是引擎支持的那几个之一**——编一个「聚类分析」出来，直接退回。
2. **列名必须在数据里真实存在**——编一个不存在的变量，直接退回。
3. **引用的原文片段必须真的出现在用户输入里**——这条最管用。要求模型说出它
   是根据哪句话判断的，如果那句话根本不在输入里，说明它在编，整条结果作废。
4. **置信度必须是 0 到 1 的数**。

任何一条不满足，都当作「没识别出来」，回退到追问。**模型的越界输出永远进不了
引擎**——有一组对抗测试专门盯着这件事。

## 置信度闸门

    >= 0.75   直接按模型的判断走，继续常规的参数抽取和缺口追问
    0.45-0.75 带着猜测追问：「你是想做信度分析吗？」
    < 0.45    不带猜测，走原来的追问

模型自报的置信度不能全信，所以再加一道：如果用户输入里连一个跟该方法沾边的
字都没有，置信度打七折。这是廉价但有效的降权。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from modules.llm import LLMClient, LLMError, LLMNotConfigured

from .registry import METHOD_SKILLS

CONFIDENCE_ACT = 0.75
CONFIDENCE_SUGGEST = 0.45
#: 用户原话里找不到任何相关字眼时的折扣
UNSUPPORTED_PENALTY = 0.7

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class Interpretation:
    """模型对一句话的翻译结果，已经过校验。"""

    method: str | None = None
    columns: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: str = ""
    reason: str = ""
    source: str = "model"
    rejected: str = ""          # 非空表示被校验挡下，内容是原因
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def should_act(self) -> bool:
        return bool(self.method) and self.confidence >= CONFIDENCE_ACT

    @property
    def should_suggest(self) -> bool:
        return bool(self.method) and CONFIDENCE_SUGGEST <= self.confidence < CONFIDENCE_ACT


def build_prompt(text: str, columns: list[str]) -> tuple[str, str]:
    catalog = "\n".join(
        f"- {key}：{spec['label']}（常见说法：{'、'.join(spec['keywords'][:4])}）"
        for key, spec in METHOD_SKILLS.items()
    )
    system = (
        "你是统计分析需求的翻译器。用户用日常说法描述想做的分析，你把它翻译成"
        "系统支持的方法标识和涉及的变量。\n\n"
        "支持的方法只有这些，不得输出其他值：\n" + catalog + "\n\n"
        "硬规则：\n"
        "1. 只输出 JSON，不要任何解释文字或代码块标记。\n"
        "2. method 必须是上面列出的标识之一；判断不了就填 null。\n"
        "3. columns 只能从给定的变量清单里选，不得杜撰。\n"
        "4. evidence 必须原样摘抄用户原话中让你做出判断的片段，不得改写。\n"
        "5. confidence 是 0 到 1 的小数，表示你对 method 判断的把握。\n"
        "6. 你只负责识别意图，不要计算任何统计量。\n\n"
        '输出格式：{"method": "...", "columns": [...], "confidence": 0.0, '
        '"evidence": "...", "reason": "..."}'
    )
    user = (
        f"变量清单：{'、'.join(columns) if columns else '（空）'}\n"
        f"用户原话：{text}"
    )
    return system, user


def interpret(
    text: str,
    columns: list[str],
    client: LLMClient | None,
) -> Interpretation:
    """让模型翻译一句话，并把结果关进笼子里。"""

    if client is None:
        return Interpretation(rejected="未配置模型")
    if not (text or "").strip():
        return Interpretation(rejected="用户输入为空")

    system, user = build_prompt(text, columns)
    try:
        response = client.complete(system=system, user=user)
    except LLMNotConfigured:
        return Interpretation(rejected="未配置模型")
    except LLMError as exc:
        return Interpretation(rejected=f"模型调用失败：{exc}")

    tokens = {"input_tokens": response.input_tokens,
              "output_tokens": response.output_tokens}

    payload = _parse_json(response.text)
    if payload is None:
        return Interpretation(rejected="模型输出不是合法 JSON", **tokens)

    return _validate(payload, text, columns, tokens)


def _parse_json(raw: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _validate(
    payload: dict[str, Any],
    text: str,
    columns: list[str],
    tokens: dict[str, int],
) -> Interpretation:
    """四道校验。任何一条不过，整条结果作废。"""

    method = payload.get("method")
    if method is None:
        return Interpretation(rejected="模型判断不出方法", **tokens)
    if method not in METHOD_SKILLS:
        return Interpretation(
            rejected=f"模型给出了引擎不支持的方法：{method}", **tokens
        )

    raw_columns = payload.get("columns") or []
    if not isinstance(raw_columns, list):
        return Interpretation(rejected="模型给出的变量清单格式不对", **tokens)
    picked = [str(c) for c in raw_columns]
    invented = [c for c in picked if c not in columns]
    if invented:
        return Interpretation(
            rejected=f"模型杜撰了数据里不存在的变量：{invented}", **tokens
        )

    evidence = str(payload.get("evidence") or "").strip()
    if not evidence:
        return Interpretation(rejected="模型没有给出判断依据", **tokens)
    if _normalize(evidence) not in _normalize(text):
        # 最管用的一道：依据不在原话里，说明在编。
        return Interpretation(
            rejected=f"模型引用的原文不存在于用户输入中：{evidence!r}", **tokens
        )

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return Interpretation(rejected="模型给出的置信度不是数字", **tokens)
    if not 0.0 <= confidence <= 1.0:
        return Interpretation(
            rejected=f"模型给出的置信度超出范围：{confidence}", **tokens
        )

    # 模型自报的把握不能全信：原话里连一个相关字眼都没有的，打折。
    if not _has_any_hint(text, method):
        confidence *= UNSUPPORTED_PENALTY

    return Interpretation(
        method=method,
        columns=picked,
        confidence=round(confidence, 4),
        evidence=evidence,
        reason=str(payload.get("reason") or "").strip(),
        source="model",
        **tokens,
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _has_any_hint(text: str, method: str) -> bool:
    """用户原话里有没有跟这个方法沾边的字。"""

    lowered = (text or "").lower()
    spec = METHOD_SKILLS[method]
    if any(keyword.lower() in lowered for keyword in spec["keywords"]):
        return True
    # 「一致性」之于信度、「关系」之于相关，这类近义表达单独列一份
    return any(hint in lowered for hint in SOFT_HINTS.get(method, ()))


#: 关键词表之外的近义说法。命中这里不算识别成功，只用于判断模型的判断合不合理。
SOFT_HINTS: dict[str, tuple[str, ...]] = {
    "reliability": ("一致", "可信", "稳定", "靠谱"),
    "validity": ("有效", "测得准", "结构"),
    "correlation": ("关系", "关联", "相关", "一起变", "影响"),
    "descriptive": ("平均", "整体情况", "分布", "概况", "基本情况"),
    "frequency": ("多少人", "占比", "比例", "各有多少"),
    "regression": ("预测", "解释", "影响因素", "决定"),
    "moderation": ("调节", "取决于", "在不同", "边界条件"),
    "parallel_mediation": ("中介", "通过", "路径", "间接"),
    "efa": ("归成几类", "维度", "structure", "几个因子"),
    "cfa": ("验证", "拟合", "模型成不成立"),
    "sem": ("模型", "路径", "整体关系"),
}
