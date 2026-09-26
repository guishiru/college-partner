"""文档校验层的测试：抓得住该抓的，也不能对合格稿误报。

校验器的失败有两种，都致命：漏报让问题溜过去，误报让人整体忽略它。所以每条
规则都配一对用例——一份触发它的坏稿，一份不该触发它的好稿。
"""

import unittest

from modules.skills.documents import (
    SEVERITY_ERROR,
    check_data_consistency,
    check_equations,
    check_heading_depth,
    check_percentage_totals,
    check_required_sections,
    check_table_totals,
    extract_heading_outline,
    headings,
    markdown_tables,
)


class HeadingTests(unittest.TestCase):
    def test_headings_are_parsed_with_their_level(self):
        text = "# 一级\n正文\n## 二级\n### 三级\n"
        self.assertEqual(headings(text), [(1, "一级"), (2, "二级"), (3, "三级")])

    def test_outline_preserves_hierarchy(self):
        text = "# 标题\n## 子节\n### 孙节\n"
        self.assertEqual(extract_heading_outline(text), "标题\n  子节\n    孙节")

    def test_depth_beyond_three_levels_is_flagged(self):
        self.assertTrue(check_heading_depth("# a\n## b\n### c\n#### d\n"))
        self.assertEqual(check_heading_depth("# a\n## b\n### c\n"), [])

    def test_required_sections_match_by_substring(self):
        text = "# 文档\n## 1 背景与目标\n## 2 范围\n"
        self.assertEqual(check_required_sections(text, ["背景与目标", "范围"]), [])
        findings = check_required_sections(text, ["背景与目标", "假设索引"])
        self.assertEqual(findings[0].details["missing"], ["假设索引"])


class ArithmeticTests(unittest.TestCase):
    def test_a_wrong_equation_is_caught(self):
        findings = check_equations("合格率 8200 / 8905 = 0.95，请知悉。")
        self.assertEqual(findings[0].code, "equation_mismatch")
        self.assertAlmostEqual(findings[0].details["actual"], 0.9208, places=3)

    def test_a_correct_equation_is_reported_as_verified(self):
        findings = check_equations("本月 4000 + 3200 + 1500 = 8700 单。")
        self.assertEqual(findings[0].code, "equations_consistent")
        self.assertEqual(findings[0].details["checked"], 1)

    def test_text_without_equations_produces_nothing(self):
        self.assertEqual(check_equations("本月工作顺利推进，无异常。"), [])

    def test_percentage_equations_are_scaled_before_comparing(self):
        self.assertEqual(check_equations("占比 1 / 4 = 25%"), [
            f for f in check_equations("占比 1 / 4 = 25%") if f.code == "equations_consistent"
        ])

    def test_a_wrong_table_total_is_caught(self):
        table = (
            "| 渠道 | 单量 |\n|---|---|\n"
            "| 门店 | 4000 |\n| 电商 | 3200 |\n| 合计 | 8905 |\n"
        )
        findings = check_table_totals(table)
        self.assertEqual(findings[0].code, "table_total_mismatch")
        self.assertEqual(findings[0].details["actual"], 7200)

    def test_a_correct_table_total_is_silent(self):
        table = (
            "| 渠道 | 单量 |\n|---|---|\n"
            "| 门店 | 4000 |\n| 电商 | 3200 |\n| 合计 | 7200 |\n"
        )
        self.assertEqual(check_table_totals(table), [])

    def test_a_table_without_a_total_row_is_not_second_guessed(self):
        table = "| 渠道 | 单量 |\n|---|---|\n| 门店 | 4000 |\n| 电商 | 3200 |\n"
        self.assertEqual(check_table_totals(table), [])

    def test_percentages_that_do_not_reach_one_hundred_are_flagged(self):
        table = (
            "| 渠道 | 占比 |\n|---|---|\n"
            "| 门店 | 44.9% |\n| 电商 | 35.9% |\n| 其他 | 16.8% |\n"
        )
        findings = check_percentage_totals(table)
        self.assertEqual(findings[0].code, "percentage_total_mismatch")

    def test_percentages_summing_to_one_hundred_are_silent(self):
        table = (
            "| 渠道 | 占比 |\n|---|---|\n"
            "| 门店 | 45.0% |\n| 电商 | 36.0% |\n| 其他 | 19.0% |\n"
        )
        self.assertEqual(check_percentage_totals(table), [])

    def test_the_same_column_is_not_reported_twice(self):
        """合计行核对和百分比加总不能对同一列各报一次。"""

        table = (
            "| 渠道 | 占比 |\n|---|---|\n"
            "| 门店 | 44.9% |\n| 电商 | 35.9% |\n| 其他 | 16.8% |\n| 合计 | 100% |\n"
        )
        codes = [f.code for f in check_data_consistency(table)]
        self.assertEqual(codes.count("table_total_mismatch"), 1)
        self.assertEqual(codes.count("percentage_total_mismatch"), 0)

    def test_markdown_tables_are_parsed_without_separator_rows(self):
        table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        self.assertEqual(markdown_tables(table), [[["a", "b"], ["1", "2"]]])

    def test_a_clean_document_produces_no_errors(self):
        clean = (
            "# 月度汇报\n\n本月产量 4000 + 3200 = 7200 件。\n\n"
            "| 渠道 | 单量 |\n|---|---|\n| 门店 | 4000 |\n| 电商 | 3200 |\n| 合计 | 7200 |\n"
        )
        errors = [f for f in check_data_consistency(clean) if f.severity == SEVERITY_ERROR]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
