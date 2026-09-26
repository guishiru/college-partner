"""用户上传的数据是不规范的，这里把「不规范到底会怎样」钉死。

这批夹具来自一次专项审查：造 17 个真实场景的脏文件跑一遍，看哪些被拦住、
哪些静默通过。当时有六处是**静默给出错误结果**——不报错、不警告，只是安静地
返回一个没有意义的数。这类问题比崩溃危险得多，因为用户会照抄进论文。

每个用例都注明期望的行为类别：

``blocked``    读取阶段直接拒绝（坏行、编码问题这类）
``error``      能读进来，但存在阻塞性问题，不允许进入分析
``warning``    可以分析，但必须把风险明确告诉用户
``clean``      正常处理，可能带可追溯的处理记录
"""

from __future__ import annotations

import unittest
from pathlib import Path

from modules.analysis.data_processing import DataProcessingError, prepare_for_analysis
from modules.analysis.engine import AnalysisEngine, AnalysisInputError

FIXTURES = Path(__file__).resolve().parent / "dirty_fixtures"

# 文件 -> (期望类别, 期望出现的问题码)
EXPECTATIONS = {
    "坏行.csv": ("blocked", None),
    "重复列名.csv": ("error", "duplicate_column_name"),
    "无表头.csv": ("error", "header_row_looks_like_data"),
    "全角字符.csv": ("error", "mixed_numeric_text"),
    "中文数字.csv": ("error", "mixed_numeric_text"),
    "Excel多余表头.xlsx": ("error", "mixed_numeric_text"),
    "Excel多工作表.xlsx": ("warning", "multiple_sheets"),
    "千分位百分号.csv": ("warning", "numeric_looking_text"),
    "空列单值列.csv": ("warning", "empty_column"),
    "BOM.csv": ("clean", None),
    "GBK编码.csv": ("clean", None),
    "分号分隔.csv": ("clean", None),
    "列名空格.csv": ("clean", None),
    "单元格空格.csv": ("clean", None),
    "缺失写法.csv": ("clean", None),
    "样本量2.csv": ("clean", None),
    "Excel数值存成文本.xlsx": ("clean", None),
}


class DirtyInputTests(unittest.TestCase):
    def report_for(self, name):
        return prepare_for_analysis(str(FIXTURES / name)).report

    def test_every_fixture_has_an_expectation(self):
        present = {f.name for f in FIXTURES.iterdir() if f.is_file()}
        self.assertEqual(
            sorted(present - set(EXPECTATIONS)),
            [],
            "新增脏数据夹具必须在 EXPECTATIONS 中写明期望行为。",
        )
        self.assertEqual(sorted(set(EXPECTATIONS) - present), [])

    def test_each_fixture_behaves_as_expected(self):
        for name, (kind, code) in sorted(EXPECTATIONS.items()):
            with self.subTest(fixture=name, expected=kind):
                if kind == "blocked":
                    with self.assertRaises(DataProcessingError):
                        self.report_for(name)
                    continue

                report = self.report_for(name)
                codes = {issue.code for issue in report.issues}
                severities = {issue.severity for issue in report.issues}

                if code is not None:
                    self.assertIn(code, codes, f"{name} 少了问题码 {code}")

                if kind == "error":
                    self.assertFalse(report.ready_for_analysis)
                    self.assertIn("error", severities)
                elif kind == "warning":
                    self.assertTrue(report.ready_for_analysis)
                    self.assertNotIn("error", severities)
                    self.assertIn("warning", severities)
                else:
                    self.assertTrue(report.ready_for_analysis)
                    self.assertEqual(report.issues, [])


class SilentlyWrongResultTests(unittest.TestCase):
    """六条静默出错的回归测试。每一条当初都返回了一个看着合理的错数。"""

    @classmethod
    def setUpClass(cls):
        cls.engine = AnalysisEngine()

    def prepared(self, name):
        return prepare_for_analysis(str(FIXTURES / name)).data

    def test_an_empty_column_no_longer_zeroes_out_the_others(self):
        """曾经：多勾一个全空列，Q1 的 30 个样本静默变成 0，全表 NaN。"""

        data = self.prepared("空列单值列.csv")

        alone = self.engine.run("descriptive", data, {"cols": ["Q1"]})
        self.assertEqual(int(alone["tables"][0]["data"]["样本量"].iloc[0]), 30)

        with self.assertRaises(AnalysisInputError) as caught:
            self.engine.run("descriptive", data, {"cols": ["空列", "Q1"]})
        self.assertIn("空列", str(caught.exception))
        self.assertIn("没有任何有效值", str(caught.exception))

    def test_a_constant_column_no_longer_produces_a_negative_alpha(self):
        """曾经：常数列参与信度算出 α = -0.163，照常输出。"""

        data = self.prepared("空列单值列.csv")
        for method in ("reliability", "correlation"):
            with self.subTest(method=method):
                with self.assertRaises(AnalysisInputError) as caught:
                    self.engine.run(method, data, {"cols": ["Q1", "常数列", "Q2"]})
                self.assertIn("常数列", str(caught.exception))
                self.assertIn("方差为 0", str(caught.exception))

    def test_a_two_row_sample_no_longer_yields_perfect_reliability(self):
        """曾经：n=2 算出 α = 1.0。"""

        data = self.prepared("样本量2.csv")
        with self.assertRaises(AnalysisInputError) as caught:
            self.engine.run("reliability", data, {"cols": ["Q1", "Q2", "Q3"]})
        self.assertIn("有效样本量", str(caught.exception))

    def test_a_multi_sheet_workbook_reads_the_data_sheet(self):
        """曾经：只读第一张「填写说明」，1 行 1 列，还报告可分析。"""

        report = prepare_for_analysis(str(FIXTURES / "Excel多工作表.xlsx")).report
        self.assertEqual(report.original_rows, 30)
        self.assertEqual(report.original_columns, 3)
        issue = next(i for i in report.issues if i.code == "multiple_sheets")
        self.assertIn("数据", issue.message)
        self.assertEqual(issue.details["selected"], "数据")

    def test_numeric_looking_text_is_flagged_but_never_guessed(self):
        """千分位和百分号必须提示，但不能替用户猜量纲。"""

        prepared = prepare_for_analysis(str(FIXTURES / "千分位百分号.csv"))
        issue = next(i for i in prepared.report.issues if i.code == "numeric_looking_text")
        self.assertIn("未自动转换", issue.message)
        self.assertEqual(issue.details["thousands_separator"], ["收入"])
        self.assertEqual(issue.details["percent_sign"], ["满意度"])
        # 仍然是分类变量，没有被偷偷转成数字
        self.assertEqual(prepared.report.column_types["收入"], "categorical")

    def test_a_headerless_file_is_refused_instead_of_losing_a_respondent(self):
        """曾经：首行数据被当成列名，静默少一个样本。"""

        report = self.report_for_headerless()
        self.assertFalse(report.ready_for_analysis)
        self.assertIn(
            "header_row_looks_like_data", {issue.code for issue in report.issues}
        )

    def report_for_headerless(self):
        return prepare_for_analysis(str(FIXTURES / "无表头.csv")).report


if __name__ == "__main__":
    unittest.main()
