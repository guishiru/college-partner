"""统计量必须以数值形式存在，不能只躺在格式化字符串里。

项目约定禁止报告模块自行计算统计量。那条规则只有在结果里真的取得到数值时才
成立——如果一个系数只以 ``'0.566(<0.001***)'`` 的形式存在，报告要么解析字符
串，要么重算，两条路都不该走。而且解析本身就拿不回原值：精度被截到三位，
``<0.001`` 和 ``0.000`` 甚至不是数值而是一个界。

这里有两组测试：

* ``StructuredStatisticsTests`` 盯着已经改好的方法，确保它们不退回去；
* ``StringEncodedDebtTests`` 登记尚未处理的部分，只允许变少，不允许变多。

登记表是刻意可见的：与其让「还有 37 处统计量只是字符串」这件事散落在各个方法
文件里没人记得，不如让它变成一份会报错的清单。
"""

from __future__ import annotations

import math
import re
import unittest

import pandas as pd

from golden.cases import CASES, load_dataset

from modules.analysis.engine import AnalysisDependencyError, AnalysisEngine

# 看起来是「被格式化成字符串的数值」：0.566、-0.025(0.742)、24.09、0.0***
FORMATTED_NUMBER = re.compile(r"^\s*[-+]?\d*\.?\d+([eE][-+]?\d+)?\s*[%*]*(\(.*\))?[\s*]*$")

# 每个方法至少要有这些数值列可取，能在任意结果表里找到即可。
REQUIRED_NUMERIC = {
    "descriptive": ["平均值", "标准差", "中位数"],
    "frequency": ["频数", "百分比(%)"],
    "reliability": ["Cronbach's α系数", "样本数"],
    "correlation": ["相关系数", "P值", "样本数"],
    "cfa": ["相关系数", "P值", "√AVE", "AVE", "CR", "标准化载荷", "Z值", "非标准估计系数"],
    "regression": ["R²", "调整R²", "F", "模型自由度", "残差自由度", "P值", "B", "Beta", "VIF"],
    "moderation": ["R²", "调整R²", "系数", "t值", "p值",
                   "F", "模型自由度", "残差自由度", "P值", "△F", "△P值"],
    "parallel_mediation": ["a", "b", "a*b中介效应", "c总效应", "P值",
                           "a(p值)", "b(p值)", "a*b (P值)", "a*b (CI下限)"],
    "sem": ["标准化系数", "非标准化系数", "标准误", "Z", "P值", "值"],
    "efa": ["特征根", "旋转后特征根", "旋转后方差解释率(%)"],
}

# 尚未处理的「统计量只以字符串存在」。分两类：
#
#   丢精度 —— 值已经被四舍五入或压成 0.0 / <0.001，解析也拿不回真值；
#   仅类型 —— 字符串里是完整精度，只是同列混了 "-" 之类的占位导致列变成文本。
#
# 两类都要改，但前者更急：报告写不出真实的 p 值。
STRING_ENCODED_DEBT = {
    # 只剩三张纯展示表。它们承载的数值都已在对应的结构化表里以完整精度存在，
    # 保留是为了前端渲染，报告模块不应从这里取数。
    ("correlation", "correlation_display_matrix"): "展示矩阵，数值见 correlation_table",
    ("cfa", "corr_table"): "展示矩阵，数值见 factor_correlation_table 与 sqrt_ave_table",
    ("sem", "fit_table"): "展示宽表（含参考标准行），数值见 fit_index_table",
}

# 本身就是文本的列，不算欠账。
TEXT_COLUMNS = {
    "显著性标记", "变量", "变量1", "变量2", "因子", "因子1", "因子2", "名称",
    "题项", "指标", "参考标准", "成分", "是否保留", "模型", "因变量", "自变量",
    "Factor", "项", "题项列表", "排序", "行索引", "样本", "检验结论", "子项",
    "检验项", "→", "分析项(显变量)", "Factor(潜变量)", "是否参考题项",
}


def numeric_columns(result) -> dict[str, pd.Series]:
    found: dict[str, pd.Series] = {}
    for table in result.get("tables") or []:
        frame = table["data"]
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                found.setdefault(str(column), frame[column])
    return found


def string_encoded_tables(result) -> set[str]:
    """哪些表里还有「本该是数值却存成字符串」的列。"""

    offenders = set()
    for table in result.get("tables") or []:
        frame = table["data"]
        for column in frame.columns:
            name = str(column)
            if name in TEXT_COLUMNS or pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = frame[column].dropna().astype(str)
            values = values[~values.isin(["-", ""])]
            if values.empty:
                continue
            hits = sum(1 for value in values if FORMATTED_NUMBER.match(value))
            if hits / len(values) > 0.8:
                offenders.add(table["key"])
    return offenders


class EngineResultTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.engine = AnalysisEngine()

    def run_method(self, method):
        try:
            return self.engine.run(method, self.data, CASES[method])
        except AnalysisDependencyError as exc:
            self.skipTest(f"依赖不可用：{exc}")


class StructuredStatisticsTests(EngineResultTestCase):
    def test_every_method_exposes_its_statistics_as_numbers(self):
        for method, required in REQUIRED_NUMERIC.items():
            with self.subTest(method=method):
                available = numeric_columns(self.run_method(method))
                missing = [name for name in required if name not in available]
                self.assertEqual(
                    missing,
                    [],
                    f"{method} 的这些统计量取不到数值：{missing}。"
                    f"当前可取：{sorted(available)}",
                )

    def test_p_values_keep_their_real_magnitude(self):
        """``<0.001`` 是一个界不是一个值，报告写不出真实的 p。"""

        for method in ("correlation", "cfa", "regression"):
            with self.subTest(method=method):
                available = numeric_columns(self.run_method(method))
                self.assertIn("P值", available)
                smallest = float(available["P值"].min())
                self.assertTrue(math.isfinite(smallest))
                self.assertLess(
                    smallest,
                    1e-6,
                    f"{method} 的最小 P 值是 {smallest}，看起来已经被截断到展示精度了。",
                )

    def test_display_text_stays_derived_from_the_numbers(self):
        result = self.run_method("regression")
        summary = numeric_columns(result)
        text = result["display"]["tables"]["regression_table"]["model_summary_row"]

        self.assertIn(f"R² = {float(summary['R²'].iloc[0]):.3f}", text)
        self.assertIn(
            f"F({int(summary['模型自由度'].iloc[0])}, "
            f"{int(summary['残差自由度'].iloc[0])})",
            text,
        )

    def test_display_matrices_have_a_structured_counterpart(self):
        expectations = {
            "correlation": ("correlation_display_matrix", "correlation_table"),
            "cfa": ("corr_table", "factor_correlation_table"),
        }
        for method, (display_key, structured_key) in expectations.items():
            with self.subTest(method=method):
                result = self.run_method(method)
                keys = {table["key"] for table in result["tables"]}
                self.assertIn(display_key, keys)
                self.assertIn(
                    structured_key,
                    keys,
                    f"{method} 只有展示矩阵，没有对应的结构化结果表。",
                )
                structured = next(
                    table["data"]
                    for table in result["tables"]
                    if table["key"] == structured_key
                )
                for column in ("相关系数", "P值"):
                    self.assertTrue(
                        pd.api.types.is_numeric_dtype(structured[column]),
                        f"{structured_key}.{column} 必须是数值列。",
                    )


class StringEncodedDebtTests(EngineResultTestCase):
    """欠账只许变少，不许变多。"""

    def test_no_new_table_encodes_statistics_as_strings(self):
        registered = {key for key, _ in STRING_ENCODED_DEBT}
        for method in sorted(CASES):
            with self.subTest(method=method):
                offenders = string_encoded_tables(self.run_method(method))
                new = sorted(
                    key for key in offenders if (method, key) not in STRING_ENCODED_DEBT
                )
                self.assertEqual(
                    new,
                    [],
                    f"{method} 新出现了把统计量存成字符串的表：{new}。"
                    f"统计量应当以数值输出，展示字符串只能是它的派生物。",
                )
                self.assertLessEqual(len(offenders), len(registered))

    def test_settled_debts_are_removed_from_the_registry(self):
        """改好了就要从清单里划掉，否则清单会慢慢失真。"""

        stale = []
        for method in sorted(CASES):
            registered = {
                key for (owner, key) in STRING_ENCODED_DEBT if owner == method
            }
            if not registered:
                continue
            try:
                offenders = string_encoded_tables(
                    self.engine.run(method, self.data, CASES[method])
                )
            except AnalysisDependencyError:
                continue
            stale.extend(f"{method}.{key}" for key in sorted(registered - offenders))
        self.assertEqual(
            stale,
            [],
            f"这些已经改好了，请从 STRING_ENCODED_DEBT 中删掉：{stale}",
        )


if __name__ == "__main__":
    unittest.main()
