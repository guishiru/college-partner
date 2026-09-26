"""生成端到端测试用的假模型回复。

真调 API 的端到端测试慢、贵、每次结果不一样，CI 里跑不了。这里的回复是刻意
设计的：骨架一份，成稿一份**故意不合格**——这样校验回路和右栏的问题清单才会
真的被走到。
"""

from __future__ import annotations

import json
from pathlib import Path

SKELETON = (
    "【需求一句话】为了让客服能一次导出整段时间的订单明细，我们要做批量导出，"
    "预期把每周 40 分钟的手工整理压到 5 分钟以内\n"
    "【做 / 不做】做：按时间区间导出 xlsx；不做：自定义字段、定时推送\n"
    "【功能点清单】FR-001 按时间区间导出\n"
    "【待确认清单】单次导出上限\n"
    "【预计篇幅】约 1200 字，15 个章节"
)

# 故意缺章节、缺「明确不做」、验收含不可观察的词——要让校验器有东西可报。
DRAFT = (
    "# 订单批量导出 PRD\n\n## 1 修订记录\n\n## 2 背景与目标\n"
    "客服每天手工整理订单明细。\n\n## 7 功能需求\n\n"
    "FR-001 按时间区间导出订单明细\n触发：点击导出\n行为：系统应生成文件\n"
    "验收：导出体验流畅\n"
)


def main() -> None:
    path = Path(__file__).resolve().parent / "_fake_replies.json"
    # 骨架 1 次 + 成稿 1 次 + 最多重写 2 次
    path.write_text(
        json.dumps([SKELETON] + [DRAFT] * 3, ensure_ascii=False), encoding="utf-8"
    )
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
