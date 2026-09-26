"""生成回路与数字防线。

这两个员工让大模型直接产出交付物，所以测试守的是同一件事：**模型编出来的
东西不能就这么发出去**。规则写在提示词里是事前祈祷，写成校验回路才是事后核对。
"""

import unittest

from modules.llm import FakeClient, LLMError
from modules.skills.documents import (
    FactSheet,
    fact_sheet_from_analysis,
    generate_with_review,
)
from modules.skills.prd_writer import draft_document as draft_prd
from modules.skills.report_writer import (
    draft_document as draft_report,
    draft_skeleton as report_skeleton,
)

BAD_REPORT = """# 七月汇报

为了贯彻质量要求，现汇报如下。

## 现状分析

本月 7200 单，预计增长 35%。
"""

GOOD_REPORT = """# 质检人力需增配一人，建议本月批复

建议十月前增配 1 名专职人员。

## 单量已超出现有人力上限

本月质检单 7200 个，人均日处理 40 单。
"""


def a_sheet():
    sheet = FactSheet()
    sheet.add(7200, "本月质检单", "job_1")
    sheet.add(40, "人均日处理", "job_1")
    return sheet


class FactSheetTests(unittest.TestCase):
    def test_numbers_from_the_sheet_pass(self):
        self.assertEqual(a_sheet().check("本月 7200 单，人均 40 单。"), [])

    def test_an_invented_number_is_caught(self):
        findings = a_sheet().check("本月 7200 单，预计提升 30%。")
        self.assertEqual(findings[0].code, "number_not_in_fact_sheet")
        self.assertIn("30", findings[0].details["numbers"])

    def test_section_numbers_and_ids_are_not_treated_as_facts(self):
        text = "## 4.2 明确不做\n\nFR-003 导出订单\n\n3. 第三条\n"
        self.assertEqual(a_sheet().check(text), [])

    def test_small_counts_are_allowed(self):
        """「分三步走」「涉及 2 个系统」不是在陈述事实。"""

        self.assertEqual(a_sheet().check("分三步走，涉及 2 个系统，共 9 人参与。"), [])

    def test_a_large_invented_number_is_still_caught(self):
        findings = a_sheet().check("涉及 5000 名用户。")
        self.assertIn("5000", findings[0].details["numbers"])

    def test_lines_marked_as_pending_are_skipped(self):
        """标注行本来就是在说「这里还没有数」。"""

        self.assertEqual(a_sheet().check("退货率 [待确认] 具体数字 999。"), [])
        self.assertEqual(a_sheet().check("增长率【待补：Q3 数字 888】"), [])

    def test_an_empty_sheet_forbids_every_real_number(self):
        sheet = FactSheet()
        self.assertTrue(sheet.check("营收 12345 万元。"))
        self.assertIn("没有可引用的事实", sheet.as_prompt())

    def test_the_prompt_carries_the_source_of_each_fact(self):
        prompt = a_sheet().as_prompt()
        self.assertIn("本月质检单：7200", prompt)
        self.assertIn("job_1", prompt)

    def test_thousands_separators_match_their_plain_form(self):
        sheet = FactSheet()
        sheet.add("8,905", "质检单", "job_1")
        self.assertEqual(sheet.check("本月 8905 单。"), [])
        self.assertEqual(sheet.check("本月 8,905 单。"), [])


class FactSheetFromAnalysisTests(unittest.TestCase):
    def test_analysis_numbers_become_citable_facts_with_provenance(self):
        import pandas as pd

        results = [{
            "method": "reliability",
            "label": "信度分析",
            "provenance": {"job_id": "job_abc"},
            "meta": {"analysis_sample_size": 180},
            "tables": [{
                "key": "summary",
                "data": pd.DataFrame([{"Cronbach's α系数": 0.8657, "样本数": 180}]),
            }],
        }]
        sheet = fact_sheet_from_analysis(results)
        prompt = sheet.as_prompt()

        self.assertIn("job_abc", prompt)
        self.assertIn("0.8657", prompt)
        self.assertEqual(sheet.check("信度 α 为 0.8657，样本 180 人。"), [])
        self.assertTrue(sheet.check("信度 α 为 0.91。"))


class ReviewLoopTests(unittest.TestCase):
    def test_a_failing_draft_is_sent_back_with_the_problem_list(self):
        client = FakeClient(replies=[BAD_REPORT, GOOD_REPORT])
        result = draft_report("写七月汇报", client=client, fact_sheet=a_sheet())

        self.assertTrue(result.ready_to_deliver)
        self.assertEqual(result.rewrites, 1)
        self.assertEqual(result.usage.calls, 2)

        second_request = client.calls[1]["user"]
        self.assertIn("没有通过交付前校验", second_request)
        self.assertIn("number_not_in_fact_sheet", second_request)
        self.assertIn(BAD_REPORT.strip()[:20], second_request)

    def test_a_clean_first_draft_costs_one_call(self):
        client = FakeClient(replies=[GOOD_REPORT])
        result = draft_report("写七月汇报", client=client, fact_sheet=a_sheet())
        self.assertEqual(result.rewrites, 0)
        self.assertEqual(result.usage.calls, 1)
        self.assertTrue(result.ready_to_deliver)

    def test_rewrites_stop_at_the_limit_and_hand_back_the_problems(self):
        """改不动时原地打转只会烧钱。到上限就停，把问题交还给用户。"""

        client = FakeClient(replies=[BAD_REPORT] * 5)
        result = draft_report("写七月汇报", client=client,
                              fact_sheet=a_sheet(), max_rewrites=2)

        self.assertEqual(result.rewrites, 2)
        self.assertEqual(result.usage.calls, 3)
        self.assertFalse(result.ready_to_deliver)
        self.assertTrue(result.unresolved)
        self.assertTrue(result.text, "半成品也要交出来，不能什么都不给")

    def test_usage_accumulates_across_rewrites(self):
        client = FakeClient(replies=[BAD_REPORT, BAD_REPORT, GOOD_REPORT])
        result = draft_report("写七月汇报", client=client, fact_sheet=a_sheet())
        self.assertEqual(result.usage.calls, 3)
        self.assertGreater(result.usage.input_tokens, 0)
        self.assertGreater(result.usage.output_tokens, 0)

    def test_a_model_failure_is_reported_not_swallowed(self):
        client = FakeClient(replies=[LLMError("网络超时")])
        result = draft_report("写七月汇报", client=client)
        self.assertFalse(result.ready_to_deliver)
        self.assertIn("模型调用失败", result.failed)

    def test_without_a_client_it_says_so_instead_of_pretending(self):
        result = draft_report("写七月汇报", client=None)
        self.assertEqual(result.failed, "未配置模型")
        self.assertEqual(result.text, "")


class SkeletonTests(unittest.TestCase):
    def test_the_skeleton_pass_does_not_run_the_reviewer(self):
        """骨架本来就不完整，拿成稿的标准去查它只会全红。"""

        client = FakeClient(replies=["【中心思想】需要增配一人\n【关键句】三条"])
        result = report_skeleton("写七月汇报", client=client)
        self.assertTrue(result.text)
        self.assertEqual(result.review.findings, [])
        self.assertEqual(result.usage.calls, 1)

    def test_the_skeleton_prompt_forbids_writing_the_full_draft(self):
        client = FakeClient(replies=["【中心思想】x"])
        report_skeleton("写七月汇报", client=client)
        self.assertIn("只出骨架", client.calls[0]["system"])


class PrdDraftTests(unittest.TestCase):
    def test_the_prd_prompt_carries_the_hard_rules(self):
        client = FakeClient(replies=["# PRD"])
        draft_prd("做个导出功能", client=client, max_rewrites=0)
        system = client.calls[0]["system"]
        self.assertIn("明确不做", system)
        self.assertIn("FR-001", system)
        self.assertIn("可观察", system)
        self.assertIn("不替用户做业务决策", system)

    def test_a_weak_prd_draft_does_not_pass(self):
        client = FakeClient(replies=["# PRD\n## 背景与目标\n随便写写。"] * 3)
        result = draft_prd("做个导出功能", client=client, max_rewrites=1)
        self.assertFalse(result.ready_to_deliver)
        self.assertIn("missing_section", {f.code for f in result.unresolved})


if __name__ == "__main__":
    unittest.main()
