import unittest

from modules.skills.data_analyst import build_analysis_plan


class DataAnalystSkillTests(unittest.TestCase):
    columns = [
        "主观规范_1",
        "主观规范_2",
        "主观规范_3",
        "社会支持",
        "使用意愿",
    ]
    column_types = {
        "主观规范_1": "numeric",
        "主观规范_2": "numeric",
        "主观规范_3": "numeric",
        "社会支持": "numeric",
        "使用意愿": "numeric",
    }

    def test_regression_without_roles_asks_for_missing_parameters(self):
        plan = build_analysis_plan("做回归", self.columns, self.column_types)

        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["missing"][0]["field"], "y")
        self.assertIn("因变量", plan["question"])

    def test_regression_extracts_roles_and_is_executable(self):
        plan = build_analysis_plan(
            "做线性回归，因变量是使用意愿，自变量是主观规范、社会支持",
            self.columns,
            self.column_types,
        )

        self.assertTrue(plan["can_execute"])
        params = plan["methods"][0]["params"]
        self.assertEqual(params["y"], "使用意愿")
        self.assertEqual(
            params["x_cols"],
            ["主观规范_1", "主观规范_2", "主观规范_3", "社会支持"],
        )

    def test_dimension_name_expands_to_item_columns(self):
        plan = build_analysis_plan(
            "做主观规范的信度分析",
            self.columns,
            self.column_types,
        )

        self.assertTrue(plan["can_execute"])
        self.assertEqual(
            plan["methods"][0]["params"]["cols"],
            ["主观规范_1", "主观规范_2", "主观规范_3"],
        )

    def test_cfa_requires_factor_structure(self):
        plan = build_analysis_plan("做 CFA", self.columns, self.column_types)

        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["missing"][0]["field"], "factors")

    def test_cfa_extracts_factor_structure(self):
        plan = build_analysis_plan(
            "做CFA。规范：主观规范_1、主观规范_2、主观规范_3",
            self.columns,
            self.column_types,
        )

        self.assertTrue(plan["can_execute"])
        self.assertEqual(
            plan["methods"][0]["params"]["factors"]["规范"],
            ["主观规范_1", "主观规范_2", "主观规范_3"],
        )

    def test_unknown_request_asks_for_method(self):
        plan = build_analysis_plan("帮我看看数据", self.columns, self.column_types)

        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["missing"][0]["field"], "method")


if __name__ == "__main__":
    unittest.main()
