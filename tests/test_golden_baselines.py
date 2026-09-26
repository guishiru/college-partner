"""Numeric regression cover for the statistical engine.

The rest of the engine tests assert shape: a result came back, it has tables,
the meta says what method ran. None of them would notice if Cronbach's α
changed from 0.87 to 0.71, or if a regression coefficient flipped sign — a
wrong statistic does not raise, it just returns a plausible number.

These tests run every method over one frozen dataset and compare the numbers
against committed baselines. Any change to a statistical result becomes a red
test with the exact value that moved.

When a change is intended:

    python tests/golden/regenerate.py <method>

then read the diff before committing it, and record why in docs/decisions/.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest

from golden.cases import (
    CASES,
    DATASET_PATH,
    DATASET_SHA256,
    PENDING,
    baseline_path,
    fingerprint,
    load_dataset,
)

from modules.analysis.engine import AnalysisDependencyError, AnalysisEngine


def engine_methods() -> set[str]:
    """Ask the engine through its public interface, not its internals."""

    return {method["key"] for method in AnalysisEngine().list_methods()}


REL_TOL = 1e-6
ABS_TOL = 1e-9
MAX_REPORTED = 12


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def diff(expected, actual, path: str = "") -> list[str]:
    """Compare two normalized results, tolerating last-bit float noise."""

    where = path or "<root>"

    if _is_number(expected) and _is_number(actual):
        if math.isclose(float(expected), float(actual), rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return []
        return [f"{where}: 基线 {expected!r} → 实际 {actual!r}"]

    if isinstance(expected, dict) and isinstance(actual, dict):
        problems = []
        for key in expected.keys() - actual.keys():
            problems.append(f"{where}.{key}: 基线有，实际结果里没有")
        for key in actual.keys() - expected.keys():
            problems.append(f"{where}.{key}: 实际结果新增，基线里没有")
        for key in sorted(expected.keys() & actual.keys(), key=str):
            problems.extend(diff(expected[key], actual[key], f"{where}.{key}"))
        return problems

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: 长度 基线 {len(expected)} → 实际 {len(actual)}"]
        problems = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            problems.extend(diff(left, right, f"{where}[{index}]"))
        return problems

    if expected == actual:
        return []
    return [f"{where}: 基线 {expected!r} → 实际 {actual!r}"]


class GoldenBaselineCoverageTests(unittest.TestCase):
    """The baseline set itself must stay honest."""

    def test_the_fixture_has_not_been_edited(self):
        digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            DATASET_SHA256,
            "黄金基线数据集被改动了。改数据等于让所有基线同时失去意义；"
            "若确实要换数据，请更新 DATASET_SHA256 并重新生成全部基线。",
        )

    def test_every_engine_method_has_a_case(self):
        missing = sorted(engine_methods() - set(CASES))
        self.assertEqual(
            missing,
            [],
            "新增统计方法必须在 tests/golden/cases.py 的 CASES 中给出参数。",
        )

    def test_no_case_points_at_a_method_that_no_longer_exists(self):
        stale = sorted(set(CASES) - engine_methods())
        self.assertEqual(stale, [], "CASES 中存在引擎已经不提供的方法。")

    def test_every_method_either_has_a_baseline_or_a_recorded_reason(self):
        uncovered = [
            method
            for method in sorted(CASES)
            if not baseline_path(method).is_file() and method not in PENDING
        ]
        self.assertEqual(
            uncovered,
            [],
            "这些方法既没有基线，也没有登记在 PENDING 里。"
            "统计方法不能悄悄失去回归保护：要么补基线，要么写明原因。",
        )

    def test_pending_entries_are_cleaned_up_once_a_baseline_exists(self):
        stale = [method for method in sorted(PENDING) if baseline_path(method).is_file()]
        self.assertEqual(
            stale,
            [],
            "这些方法已经有基线了，请把它们从 PENDING 中删掉。",
        )


class GoldenBaselineTests(unittest.TestCase):
    """Every committed baseline must still reproduce."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.engine = AnalysisEngine()

    def test_results_match_the_committed_baselines(self):
        for method in sorted(CASES):
            path = baseline_path(method)
            with self.subTest(method=method):
                if not path.is_file():
                    self.skipTest(f"未生成基线：{PENDING.get(method, '原因未登记')}")
                expected = json.loads(path.read_text(encoding="utf-8"))
                try:
                    result = self.engine.run(method, self.data, CASES[method])
                except AnalysisDependencyError as exc:
                    self.skipTest(f"依赖不可用，无法验证 {method}：{exc}")

                problems = diff(expected, fingerprint(result))
                if problems:
                    shown = "\n".join(f"  {item}" for item in problems[:MAX_REPORTED])
                    more = (
                        f"\n  …… 另有 {len(problems) - MAX_REPORTED} 处差异"
                        if len(problems) > MAX_REPORTED
                        else ""
                    )
                    self.fail(
                        f"{method} 的结果与基线不一致（共 {len(problems)} 处）：\n"
                        f"{shown}{more}\n"
                        f"若这是有意的改动，运行 python tests/golden/regenerate.py {method} "
                        f"并在提交前逐行核对 diff。"
                    )

    def test_running_the_same_method_twice_gives_the_same_numbers(self):
        """A method that is not reproducible cannot be pinned at all."""

        for method in sorted(CASES):
            with self.subTest(method=method):
                try:
                    first = self.engine.run(method, self.data, CASES[method])
                    second = self.engine.run(method, self.data, CASES[method])
                except AnalysisDependencyError as exc:
                    self.skipTest(f"依赖不可用：{exc}")
                self.assertEqual(
                    diff(fingerprint(first), fingerprint(second)),
                    [],
                    f"{method} 两次运行结果不同，说明其中存在未固定的随机性。",
                )


if __name__ == "__main__":
    unittest.main()
