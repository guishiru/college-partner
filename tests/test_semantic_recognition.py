"""语义识别层：关键词优先、置信度闸门，以及模型乱说时的防线。

这一层是整个产品里**唯一让大模型影响执行路径**的地方，所以它的测试重点不是
「识别得准不准」（那取决于模型，测不了），而是「模型乱说的时候会不会出事」。

下面 ``AdversarialModelTests`` 里的每一条，都是一种模型可能的胡说。要求一律
相同：退回追问，绝不把越界的东西送进引擎。
"""

import json
import unittest

from modules.llm import FakeClient, LLMError, NullClient
from modules.skills.data_analyst import build_analysis_plan
from modules.skills.data_analyst.semantic import (
    CONFIDENCE_ACT,
    CONFIDENCE_SUGGEST,
    interpret,
)

COLUMNS = ["社会支持1", "社会支持2", "社会支持3", "主观规范1", "使用意愿"]


def reply(method="reliability", confidence=0.9, evidence="一不一致",
          columns=None, reason="测试", **extra):
    payload = {
        "method": method,
        "columns": columns if columns is not None else [],
        "confidence": confidence,
        "evidence": evidence,
        "reason": reason,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


class KeywordFirstTests(unittest.TestCase):
    """关键词命中时比模型又快又准又免费，没有理由去调它。"""

    def test_a_keyword_hit_never_calls_the_model(self):
        client = FakeClient(replies=[])      # 一旦被调用就会抛错
        plan = build_analysis_plan(
            "做信度分析，题项是 社会支持1、社会支持2、社会支持3", COLUMNS, client=client
        )
        self.assertEqual(client.calls, [])
        self.assertEqual(plan["recognition"]["source"], "keyword")
        self.assertEqual(plan["recognition"]["confidence"], 1.0)
        self.assertTrue(plan["can_execute"])

    def test_without_a_client_behaviour_is_unchanged(self):
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS)
        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["recognition"]["source"], "keyword")

    def test_a_null_client_degrades_to_asking_not_to_crashing(self):
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS, client=NullClient())
        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["recognition"]["rejected"], "未配置模型")


class ConfidenceGateTests(unittest.TestCase):
    def test_high_confidence_goes_straight_to_the_method(self):
        client = FakeClient(replies=[reply(confidence=0.9)])
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS, client=client)
        self.assertEqual([m["method"] for m in plan["methods"]], ["reliability"])
        self.assertEqual(plan["recognition"]["source"], "model")

    def test_middling_confidence_asks_with_the_guess_attached(self):
        client = FakeClient(replies=[reply(confidence=0.6)])
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS, client=client)
        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["missing"][0]["suggested_method"], "reliability")
        self.assertIn("信度分析", plan["question"])

    def test_low_confidence_asks_without_guessing(self):
        client = FakeClient(replies=[reply(confidence=0.2)])
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS, client=client)
        self.assertNotIn("suggested_method", plan["missing"][0])
        self.assertIn("请确认要做哪一种统计分析", plan["question"])

    def test_confidence_is_discounted_when_nothing_in_the_text_supports_it(self):
        """模型自报的把握不能全信：原话里毫无线索时打折。"""

        supported = interpret(
            "看看这几道题一不一致", COLUMNS,
            FakeClient(replies=[reply(confidence=0.9, evidence="一不一致")]),
        )
        unsupported = interpret(
            "随便帮我弄一下吧", COLUMNS,
            FakeClient(replies=[reply(confidence=0.9, evidence="随便帮我弄一下吧")]),
        )
        self.assertEqual(supported.confidence, 0.9)
        self.assertLess(unsupported.confidence, 0.9)

    def test_the_thresholds_are_ordered(self):
        self.assertLess(CONFIDENCE_SUGGEST, CONFIDENCE_ACT)


class AdversarialModelTests(unittest.TestCase):
    """模型乱说的每一种方式，结果都必须是退回追问。"""

    def assert_rejected(self, model_reply, text="看看这几道题一不一致", contains=""):
        client = FakeClient(replies=[model_reply])
        result = interpret(text, COLUMNS, client)
        self.assertIsNone(result.method, f"不该被采纳：{model_reply!r}")
        self.assertTrue(result.rejected, "必须记录退回原因")
        if contains:
            self.assertIn(contains, result.rejected)

        plan = build_analysis_plan(text, COLUMNS, client=FakeClient(replies=[model_reply]))
        self.assertFalse(plan["can_execute"], "越界的模型输出不得进入执行")
        return result

    def test_a_method_the_engine_does_not_support_is_refused(self):
        self.assert_rejected(reply(method="聚类分析"), contains="不支持的方法")

    def test_invented_column_names_are_refused(self):
        self.assert_rejected(
            reply(columns=["社会支持1", "不存在的变量"]), contains="杜撰"
        )

    def test_fabricated_evidence_is_refused(self):
        """最管用的一道：引用的原话根本不在输入里。"""

        self.assert_rejected(
            reply(evidence="请帮我做一下信度分析"), contains="引用的原文不存在"
        )

    def test_missing_evidence_is_refused(self):
        self.assert_rejected(reply(evidence=""), contains="没有给出判断依据")

    def test_confidence_out_of_range_is_refused(self):
        self.assert_rejected(reply(confidence=3.0), contains="超出范围")
        self.assert_rejected(reply(confidence=-1), contains="超出范围")

    def test_confidence_that_is_not_a_number_is_refused(self):
        self.assert_rejected(reply(confidence="很有把握"), contains="不是数字")

    def test_a_non_json_answer_is_refused(self):
        self.assert_rejected("我觉得你应该做个信度分析。", contains="不是合法 JSON")

    def test_broken_json_is_refused(self):
        self.assert_rejected('{"method": "reliability", ', contains="不是合法 JSON")

    def test_a_null_method_is_refused(self):
        self.assert_rejected(reply(method=None), contains="判断不出方法")

    def test_a_malformed_column_list_is_refused(self):
        self.assert_rejected(reply(columns="社会支持1"), contains="格式不对")

    def test_a_model_failure_degrades_to_asking(self):
        client = FakeClient(replies=[LLMError("网络超时")])
        result = interpret("看看这几道题一不一致", COLUMNS, client)
        self.assertIsNone(result.method)
        self.assertIn("模型调用失败", result.rejected)

    def test_prose_wrapped_json_is_still_accepted(self):
        """模型爱加代码块标记和客套话，这个不算越界，要能容忍。"""

        wrapped = "好的，分析如下：\n```json\n" + reply() + "\n```\n希望有帮助。"
        result = interpret("看看这几道题一不一致", COLUMNS, FakeClient(replies=[wrapped]))
        self.assertEqual(result.method, "reliability")


class UsageAccountingTests(unittest.TestCase):
    def test_token_usage_is_carried_back_for_accounting(self):
        client = FakeClient(replies=[reply()])
        plan = build_analysis_plan("看看这几道题一不一致", COLUMNS, client=client)
        recognition = plan["recognition"]
        self.assertGreater(recognition["input_tokens"], 0)
        self.assertGreater(recognition["output_tokens"], 0)

    def test_a_keyword_hit_costs_nothing(self):
        plan = build_analysis_plan("做信度分析，题项是 社会支持1、社会支持2", COLUMNS,
                                   client=FakeClient(replies=[]))
        self.assertEqual(plan["recognition"]["input_tokens"], 0)
        self.assertEqual(plan["recognition"]["output_tokens"], 0)


class PromptTests(unittest.TestCase):
    def test_the_prompt_lists_only_supported_methods_and_real_columns(self):
        from modules.skills.data_analyst.semantic import build_prompt
        from modules.skills.data_analyst.registry import METHOD_SKILLS

        system, user = build_prompt("随便问问", COLUMNS)
        for key in METHOD_SKILLS:
            self.assertIn(key, system)
        for column in COLUMNS:
            self.assertIn(column, user)
        self.assertIn("不要计算任何统计量", system)


if __name__ == "__main__":
    unittest.main()
