"""PRD 撰写与汇报文档两个 Skill 的确定性层。

两份样稿贯穿全文：一份合格的、一份典型有问题的。合格稿必须一条 error 都不报
——校验器一旦对好稿误报，人就会开始整体忽略它，那时漏报也不会有人发现。
"""

import unittest

from modules.skills.documents import SEVERITY_ERROR
from modules.skills.prd_writer import (
    COMPLETENESS_FIELDS,
    REQUIRED_SECTIONS,
    completeness_gate,
    review_prd,
)
from modules.skills.report_writer import (
    VERDICTS,
    review_report,
    verdict_is_consistent,
)

GOOD_PRD = """# 订单批量导出 PRD

## 1 修订记录
| 版本 | 日期 | 修改人 | 改了什么 |
|---|---|---|---|
| v0.1 | 2026-09-24 | 桂仕如 | 初稿 |

## 2 背景与目标
客服每天手工整理订单明细约 40 分钟。不做的话，单量翻倍后这项工作会占掉一个人力。

## 3 名词表
- 订单：已支付且未取消的交易记录。
- 活跃客服：近 30 天内处理过至少 1 单的客服账号。

## 4 范围
### 4.1 本期做
按时间区间导出订单明细为 Excel。
### 4.2 明确不做
不做自定义字段选择，不做定时推送，不做导出结果的二次编辑。

## 5 用户与场景
客服主管，每周一上午导出上周明细做周会材料，目前靠手工复制粘贴。

## 6 主流程
选择时间区间 → 点击导出 → 后台生成 → 完成后下载。
分支：单量超过上限时提示缩小区间。

## 7 功能需求
FR-001 按时间区间导出订单明细
触发：用户在订单列表页选定时间区间并点击「导出」
行为：系统应生成 xlsx 文件并提供下载链接
验收：当选定区间内有 1000 条订单，若点击导出，则 10 秒内返回下载链接，文件行数为 1000

FR-002 导出量超限时阻止
触发：选定区间内订单数超过 10000 条
行为：系统应拒绝导出并提示缩小区间
验收：当区间内有 10001 条订单，若点击导出，则不生成文件并显示「单次导出上限 10000 条」

## 8 异常与边界
- 空态：区间内没有订单时，提示「该区间无订单」，不生成文件。
- 错误态：生成失败时保留任务记录并提示重试。接口超时按失败处理。
- 加载态：超过 3 秒显示进度提示。
- 权限：无导出权限的账号看不到导出按钮。
- 重复与并发：同一账号同时只允许一个导出任务，连点第二次提示「已有任务进行中」。
- 极端值：0 条走空态；10 万条走超限拒绝；超长备注截断至 500 字。
- 存量数据：上线前的历史订单同样可导出，不做数据迁移。
- 可逆性：导出为只读操作，不涉及撤销。

## 9 数据需求
字段：订单号、下单时间、金额、状态。来源：订单库主表。口径见名词表。埋点：导出点击与成功各一个事件。

## 10 非功能需求
单次导出上限 10000 条，生成耗时 P95 < 10s。

## 11 依赖与风险
依赖订单库读库。[假设] 读库可承受每日 50 次全量区间查询。

## 12 发布策略
灰度 10% 客服账号，开关可一键关闭，回滚不涉及数据变更。

## 13 成功指标
主指标：客服周均手工整理时长从 40 分钟降到 5 分钟以内。
反向指标：订单库读库 P95 延迟不上升超过 10%。

## 14 待确认清单
暂无。

## 15 假设索引
| # | 假设 | 所在章节 | 如果这条是错的会怎样 | 怎么验证 |
|---|---|---|---|---|
| 1 | 读库可承受每日 50 次全量区间查询 | 11 依赖与风险 | 导出会拖慢线上查询 | 找 DBA 要读库当前负载 |
"""

BAD_PRD = """# 导出功能 PRD

## 1 修订记录
## 2 背景与目标
## 3 名词表
## 4 范围
本期做导出。
## 5 用户与场景
## 6 主流程
## 7 功能需求
FR-001 用户可以导出
触发：点击按钮
行为：系统应导出
验收：导出体验流畅，速度尽量快
## 8 异常与边界
失败时报错。
## 9 数据需求
## 10 非功能需求
系统应具备良好的性能和稳定性。
## 11 依赖与风险
[假设] 单量不大。
## 12 发布策略
## 13 成功指标
## 14 待确认清单
## 15 假设索引
| # | 假设 | 所在章节 | 如果错了会怎样 | 怎么验证 |
|---|---|---|---|---|
"""

GOOD_REPORT = """# 质检人力需要增配一人，建议本月批复

建议在十月前为质检组增配 1 名专职人员，否则四季度旺季的质检时效将从 2 天延长到 5 天以上。

## 单量已超出现有人力的处理上限

本月质检单 7200 个，按现有 3 人、人均日处理 40 单计算，月产能上限为 3600 + 2400 = 6000 单。

| 渠道 | 单量 |
|---|---|
| 门店 | 4000 |
| 电商 | 3200 |
| 合计 | 7200 |

## 加班已连续三个月，不可持续

九月质检组人均加班 32 小时，为公司平均值的两倍。

## 增配一人可把时效拉回两天

按人均日处理 40 单，增配后月产能为 8000 单，高于当前单量。
"""

BAD_REPORT = """# 七月质检工作汇报

为了贯彻年度质量管理要求，现将七月质检工作情况汇报如下。

## 现状分析

本月共处理质检单 8905 个，合格率 8200 / 8905 = 0.95。

| 渠道 | 单量 |
|---|---|
| 门店 | 4000 |
| 电商 | 3200 |
| 合计 | 8905 |

## 存在的问题

退货率 【待补：Q3 退货率具体数字】。

## 下一步

### 细化措施

#### 更细的一层

##### 再细一层
"""


class PrdReviewTests(unittest.TestCase):
    def test_a_sound_prd_passes_the_gate(self):
        review = review_prd(GOOD_PRD)
        errors = [f"{f.code}: {f.message}" for f in review.by_severity(SEVERITY_ERROR)]
        self.assertEqual(errors, [], "合格 PRD 不应产生任何 error")
        self.assertTrue(review.ready_to_deliver)

    def test_a_weak_prd_is_stopped(self):
        review = review_prd(BAD_PRD)
        self.assertFalse(review.ready_to_deliver)
        expected = {
            "scope_missing_exclusions",
            "acceptance_not_observable",
            "exception_coverage_incomplete",
            "non_functional_without_threshold",
            "assumption_index_incomplete",
        }
        self.assertTrue(
            expected <= review.codes(),
            f"漏了：{sorted(expected - review.codes())}",
        )

    def test_missing_sections_are_named(self):
        review = review_prd("# 空文档\n## 背景与目标\n")
        finding = next(f for f in review.findings if f.code == "missing_section")
        self.assertIn("假设索引", finding.details["missing"])
        self.assertIn("功能需求", finding.details["missing"])

    def test_a_document_without_numbered_requirements_is_refused(self):
        text = GOOD_PRD.replace("FR-001", "需求一").replace("FR-002", "需求二")
        self.assertIn("no_functional_requirements", review_prd(text).codes())

    def test_every_required_section_is_declared(self):
        self.assertEqual(len(REQUIRED_SECTIONS), 15)
        self.assertIn("假设索引", REQUIRED_SECTIONS)
        self.assertIn("异常与边界", REQUIRED_SECTIONS)


class CompletenessGateTests(unittest.TestCase):
    def all_fields(self, value="green"):
        return {name: value for name in COMPLETENESS_FIELDS}

    def test_all_green_passes(self):
        self.assertTrue(completeness_gate(self.all_fields()).ready_to_deliver)

    def test_a_single_red_blocks_generation(self):
        fields = self.all_fields()
        fields["数据口径"] = "red"
        review = completeness_gate(fields)
        self.assertFalse(review.ready_to_deliver)
        self.assertIn("数据口径", review.findings[0].details["fields"])

    def test_yellow_warns_without_blocking(self):
        fields = self.all_fields()
        fields["主流程"] = "yellow"
        review = completeness_gate(fields)
        self.assertTrue(review.ready_to_deliver)
        self.assertIn("gate_yellow", review.codes())

    def test_skipping_a_field_is_itself_a_failure(self):
        fields = self.all_fields()
        fields.pop("验收标准")
        self.assertIn("gate_not_evaluated", completeness_gate(fields).codes())

    def test_unknown_fields_are_refused(self):
        with self.assertRaises(ValueError):
            completeness_gate({"不存在的项": "green"})


class ReportReviewTests(unittest.TestCase):
    def test_a_sound_report_passes(self):
        review = review_report(GOOD_REPORT)
        problems = [
            f"{f.code}: {f.message}"
            for f in review.findings
            if f.severity in ("error", "warning")
        ]
        self.assertEqual(problems, [], "合格汇报稿不应产生 error 或 warning")

    def test_a_weak_report_is_diagnosed(self):
        review = review_report(BAD_REPORT)
        expected = {
            "equation_mismatch",
            "table_total_mismatch",
            "conclusion_buried",
            "no_ask",
            "heading_without_information",
            "heading_too_deep",
            "placeholder_not_filled",
        }
        self.assertTrue(
            expected <= review.codes(),
            f"漏了：{sorted(expected - review.codes())}",
        )
        self.assertFalse(review.ready_to_deliver)

    def test_the_verdict_must_match_the_problem_list(self):
        findings = review_report(BAD_REPORT).findings
        self.assertFalse(verdict_is_consistent("可直接发", findings))
        self.assertFalse(verdict_is_consistent("局部调整", findings))
        self.assertTrue(verdict_is_consistent("需动主干", findings))

    def test_a_clean_report_may_be_sent_as_is(self):
        findings = review_report(GOOD_REPORT).findings
        self.assertTrue(verdict_is_consistent("可直接发", findings))

    def test_an_invalid_verdict_is_refused(self):
        with self.assertRaises(ValueError):
            verdict_is_consistent("挺好的", [])
        self.assertEqual(VERDICTS, ("可直接发", "局部调整", "需动主干"))


class SkillBoundaryTests(unittest.TestCase):
    """校验器只查形式，不做判断——越界就会产生说不清的误报。"""

    def test_the_checkers_do_not_judge_wording(self):
        """同样的结构、不同的文风，结论必须一致。"""

        formal = GOOD_REPORT
        casual = GOOD_REPORT.replace("建议", "我建议").replace("否则", "不然")
        self.assertEqual(
            {f.code for f in review_report(formal).findings},
            {f.code for f in review_report(casual).findings},
        )

    def test_external_facts_are_never_challenged(self):
        """只报「对不上」，不报「数字本身对不对」。"""

        text = "# 汇报\n\n建议批复。本月质检单 999999999 个，1 + 1 = 2。\n"
        codes = {f.code for f in review_report(text).findings}
        self.assertNotIn("equation_mismatch", codes)


if __name__ == "__main__":
    unittest.main()
