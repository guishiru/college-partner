"""模型用量记账。

用量是横切的东西：加字段容易，补历史不可能。所以这些测试守的不只是「数加得
对不对」，还有「会不会漏记」和「会不会串用户」。
"""

import tempfile
import unittest
import uuid
from pathlib import Path

from modules.jobs import JobError, JobStore


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.directory.name) / "state" / "jobs.sqlite")
        self.alice = uuid.uuid4().hex
        self.bob = uuid.uuid4().hex
        self.session = str(uuid.uuid4())

    def tearDown(self):
        self.directory.cleanup()

    def a_job(self, user_id=None, session_id=None, skill="data_analyst.analyze", key=None):
        job, _ = self.store.start(
            user_id=user_id or self.alice,
            session_id=session_id or self.session,
            employee="data_analyst",
            skill=skill,
            input_ref={},
            idempotency_key=key or uuid.uuid4().hex,
        )
        return job

    def test_a_new_job_starts_with_zero_usage(self):
        job = self.a_job()
        self.assertEqual((job.llm_calls, job.input_tokens, job.output_tokens), (0, 0, 0))
        self.assertIsNone(job.model)

    def test_usage_accumulates_instead_of_overwriting(self):
        """一个任务可能调多次模型：识别一次、成稿一次、重写一次。"""

        job = self.a_job()
        self.store.record_usage(job.job_id, model="qwen-plus", input_tokens=800, output_tokens=90)
        self.store.record_usage(job.job_id, model="qwen-plus", input_tokens=1500, output_tokens=300)
        updated = self.store.record_usage(
            job.job_id, model="qwen-plus", input_tokens=200, output_tokens=50
        )

        self.assertEqual(updated.llm_calls, 3)
        self.assertEqual(updated.input_tokens, 2500)
        self.assertEqual(updated.output_tokens, 440)
        self.assertEqual(updated.total_tokens, 2940)
        self.assertEqual(updated.model, "qwen-plus")

    def test_usage_is_recorded_even_when_the_job_fails(self):
        """钱已经花了，任务失不失败都得记账。"""

        job = self.a_job()
        self.store.record_usage(job.job_id, model="qwen-plus", input_tokens=500, output_tokens=60)
        self.store.fail(job.job_id, "引擎报错")

        failed = self.store.get(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.total_tokens, 560)

    def test_a_retry_keeps_accumulating(self):
        job = self.a_job(key="same")
        self.store.record_usage(job.job_id, model="qwen-plus", input_tokens=500, output_tokens=60)
        self.store.fail(job.job_id, "超时")
        retried, reused = self.store.start(
            user_id=self.alice, session_id=self.session, employee="data_analyst",
            skill="data_analyst.analyze", input_ref={}, idempotency_key="same",
        )
        self.assertFalse(reused)
        self.store.record_usage(retried.job_id, model="qwen-plus", input_tokens=500, output_tokens=60)

        self.assertEqual(self.store.get(job.job_id).total_tokens, 1120)

    def test_negative_usage_is_refused(self):
        job = self.a_job()
        with self.assertRaises(JobError):
            self.store.record_usage(job.job_id, model="m", input_tokens=-1, output_tokens=0)

    def test_recording_against_an_unknown_job_fails(self):
        with self.assertRaises(JobError):
            self.store.record_usage(uuid.uuid4().hex, model="m", input_tokens=1, output_tokens=1)

    def test_a_keyword_hit_records_nothing(self):
        """关键词命中时没调模型，不该凭空多出调用次数。"""

        job = self.a_job()
        self.assertEqual(self.store.get(job.job_id).llm_calls, 0)


class UsageRollupTests(UsageAccountingTests):
    def test_session_usage_sums_every_job(self):
        for tokens in (100, 200, 300):
            job = self.a_job()
            self.store.record_usage(job.job_id, model="qwen-plus",
                                    input_tokens=tokens, output_tokens=10)

        usage = self.store.usage_for_session(self.session, self.alice)
        self.assertEqual(usage["jobs"], 3)
        self.assertEqual(usage["llm_calls"], 3)
        self.assertEqual(usage["input_tokens"], 600)
        self.assertEqual(usage["total_tokens"], 630)

    def test_usage_is_broken_down_by_skill(self):
        for skill, tokens in (("data_analyst.analyze", 1000), ("data_analyst.inspect", 200)):
            job = self.a_job(skill=skill)
            self.store.record_usage(job.job_id, model="qwen-plus",
                                    input_tokens=tokens, output_tokens=0)

        by_skill = self.store.usage_for_session(self.session, self.alice)["by_skill"]
        self.assertEqual(by_skill[0]["skill"], "data_analyst.analyze")   # 按用量降序
        self.assertEqual(by_skill[0]["total_tokens"], 1000)
        self.assertEqual(by_skill[1]["total_tokens"], 200)

    def test_one_user_never_sees_another_users_usage(self):
        mine = self.a_job(user_id=self.alice)
        theirs = self.a_job(user_id=self.bob)
        self.store.record_usage(mine.job_id, model="m", input_tokens=100, output_tokens=0)
        self.store.record_usage(theirs.job_id, model="m", input_tokens=9999, output_tokens=0)

        self.assertEqual(self.store.usage_for_user(self.alice)["total_tokens"], 100)
        self.assertEqual(self.store.usage_for_user(self.bob)["total_tokens"], 9999)
        self.assertEqual(
            self.store.usage_for_session(self.session, self.alice)["total_tokens"], 100
        )

    def test_user_usage_spans_sessions(self):
        """限额要按用户算，不能被新建会话绕过。"""

        first = self.a_job(session_id=str(uuid.uuid4()))
        second = self.a_job(session_id=str(uuid.uuid4()))
        for job in (first, second):
            self.store.record_usage(job.job_id, model="m", input_tokens=500, output_tokens=0)

        self.assertEqual(self.store.usage_for_user(self.alice)["total_tokens"], 1000)

    def test_a_user_with_no_jobs_reads_zero_not_an_error(self):
        usage = self.store.usage_for_user(uuid.uuid4().hex)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["by_skill"], [])


if __name__ == "__main__":
    unittest.main()
