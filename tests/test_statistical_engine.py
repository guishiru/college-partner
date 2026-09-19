import unittest

import numpy as np
import pandas as pd

from modules.analysis.engine import AnalysisEngine, AnalysisInputError


class StatisticalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260919)
        n = 140
        f1 = rng.normal(size=n)
        f2 = 0.55 * f1 + rng.normal(scale=0.85, size=n)
        x = rng.normal(size=n)
        m = 0.45 * x + rng.normal(scale=0.9, size=n)
        y = 0.35 * x + 0.4 * m + rng.normal(scale=0.8, size=n)

        cls.data = pd.DataFrame(
            {
                "F1_1": f1 + rng.normal(scale=0.35, size=n),
                "F1_2": f1 + rng.normal(scale=0.35, size=n),
                "F1_3": f1 + rng.normal(scale=0.35, size=n),
                "F2_1": f2 + rng.normal(scale=0.35, size=n),
                "F2_2": f2 + rng.normal(scale=0.35, size=n),
                "F2_3": f2 + rng.normal(scale=0.35, size=n),
                "X": x,
                "M": m,
                "Y": y,
                "Group": np.where(x > 0, "高", "低"),
            }
        )
        cls.engine = AnalysisEngine()
        cls.factors = {
            "F1": ["F1_1", "F1_2", "F1_3"],
            "F2": ["F2_1", "F2_2", "F2_3"],
        }

    def assert_result(self, result, method):
        self.assertEqual(result["method"], method)
        self.assertTrue(result["tables"])
        self.assertEqual(result["meta"]["method_key"], method)
        self.assertEqual(result["meta"]["method_version"], "v517-validation-baseline")
        for table in result["tables"]:
            self.assertIsInstance(table["data"], pd.DataFrame)

    def test_descriptive(self):
        result = self.engine.run(
            "descriptive", self.data, {"cols": ["F1_1", "F1_2"]}
        )
        self.assert_result(result, "descriptive")

    def test_frequency(self):
        result = self.engine.run("frequency", self.data, {"cols": ["Group"]})
        self.assert_result(result, "frequency")

    def test_reliability(self):
        result = self.engine.run(
            "reliability", self.data, {"cols": self.factors["F1"]}
        )
        self.assert_result(result, "reliability")
        summary = result["tables"][0]["data"]
        self.assertIn("Cronbach's α系数", summary.columns)

    def test_validity(self):
        result = self.engine.run("validity", self.data, {"cols": self.factors["F1"]})
        self.assert_result(result, "validity")

    def test_correlation(self):
        result = self.engine.run(
            "correlation", self.data, {"cols": ["X", "M", "Y"]}
        )
        self.assert_result(result, "correlation")

    def test_efa(self):
        result = self.engine.run(
            "efa",
            self.data,
            {"cols": [*self.factors["F1"], *self.factors["F2"]], "n_factors": 2},
        )
        self.assert_result(result, "efa")

    def test_cfa(self):
        result = self.engine.run("cfa", self.data, {"factors": self.factors})
        self.assert_result(result, "cfa")

    def test_regression(self):
        result = self.engine.run(
            "regression", self.data, {"y": "Y", "x_cols": ["X", "M"]}
        )
        self.assert_result(result, "regression")

    def test_moderation(self):
        result = self.engine.run(
            "moderation",
            self.data,
            {"x": "X", "y": "Y", "m": "M"},
        )
        self.assert_result(result, "moderation")

    def test_parallel_mediation(self):
        result = self.engine.run(
            "parallel_mediation",
            self.data,
            {"x": "X", "y": "Y", "m_cols": ["M"], "n_boot": 80},
        )
        self.assert_result(result, "parallel_mediation")

    def test_sem(self):
        result = self.engine.run(
            "sem",
            self.data,
            {"factors": self.factors, "paths": [("F1", "F2")]},
        )
        self.assert_result(result, "sem")

    def test_regression_requires_dependent_variable(self):
        with self.assertRaises(AnalysisInputError):
            self.engine.run("regression", self.data, {"x_cols": ["X"]})

    def test_numeric_methods_reject_text_without_preprocessing(self):
        data = self.data.copy()
        data["F1_1"] = data["F1_1"].map(lambda value: f" {value} ")
        with self.assertRaises(AnalysisInputError):
            self.engine.run(
                "reliability", data, {"cols": self.factors["F1"]}
            )


if __name__ == "__main__":
    unittest.main()
