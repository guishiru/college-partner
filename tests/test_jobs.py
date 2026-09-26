import tempfile
import unittest
import uuid
from pathlib import Path

from modules.jobs import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    JobAccessError,
    JobConflictError,
    JobError,
    JobStore,
)


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.directory.name) / "state" / "workbench.sqlite")
        self.alice = uuid.uuid4().hex
        self.bob = uuid.uuid4().hex
        self.session_id = str(uuid.uuid4())

    def tearDown(self):
        self.directory.cleanup()

    def start(self, user_id=None, key="analysis-1", session_id=None):
        return self.store.start(
            user_id=user_id or self.alice,
            session_id=session_id or self.session_id,
            employee="data_analyst",
            skill="data_analyst.analyze",
            input_ref={"file_id": "abc", "requirement": "做信度分析"},
            idempotency_key=key,
        )

    def test_a_new_job_starts_running_and_records_its_input(self):
        job, reused = self.start()

        self.assertFalse(reused)
        self.assertEqual(job.status, STATUS_RUNNING)
        self.assertEqual(job.user_id, self.alice)
        self.assertEqual(job.session_id, self.session_id)
        self.assertEqual(job.attempts, 1)
        self.assertEqual(job.input_ref["requirement"], "做信度分析")

    def test_job_must_carry_user_session_and_key(self):
        with self.assertRaises(JobError):
            self.store.start(
                user_id="",
                session_id=self.session_id,
                employee="data_analyst",
                skill="s",
                input_ref={},
                idempotency_key="k",
            )
        with self.assertRaises(JobError):
            self.store.start(
                user_id=self.alice,
                session_id=self.session_id,
                employee="data_analyst",
                skill="s",
                input_ref={},
                idempotency_key="",
            )

    def test_a_second_identical_request_is_refused_while_running(self):
        self.start()
        with self.assertRaises(JobConflictError):
            self.start()

    def test_a_succeeded_job_is_returned_instead_of_running_again(self):
        job, _ = self.start()
        self.store.succeed(job.job_id, {"report_id": "r1"})

        again, reused = self.start()

        self.assertTrue(reused)
        self.assertEqual(again.job_id, job.job_id)
        self.assertEqual(again.status, STATUS_SUCCEEDED)
        self.assertEqual(again.result_ref, {"report_id": "r1"})
        self.assertEqual(again.attempts, 1)

    def test_a_failed_job_can_be_retried_in_place(self):
        job, _ = self.start()
        self.store.fail(job.job_id, "引擎依赖缺失")
        self.assertEqual(self.store.get(job.job_id).status, STATUS_FAILED)

        retried, reused = self.start()

        self.assertFalse(reused)
        self.assertEqual(retried.job_id, job.job_id)
        self.assertEqual(retried.status, STATUS_RUNNING)
        self.assertEqual(retried.attempts, 2)
        self.assertIsNone(retried.failure_reason)

    def test_a_different_request_gets_its_own_job(self):
        first, _ = self.start(key="analysis-1")
        self.store.succeed(first.job_id, {"report_id": "r1"})

        second, reused = self.start(key="analysis-2")

        self.assertFalse(reused)
        self.assertNotEqual(second.job_id, first.job_id)

    def test_the_same_key_in_another_session_is_a_separate_job(self):
        first, _ = self.start()
        other_session = str(uuid.uuid4())

        second, _ = self.start(session_id=other_session)

        self.assertNotEqual(second.job_id, first.job_id)

    def test_another_user_cannot_touch_the_job(self):
        job, _ = self.start()
        self.store.succeed(job.job_id, {"report_id": "r1"})

        with self.assertRaises(JobAccessError):
            self.start(user_id=self.bob)
        with self.assertRaises(JobAccessError):
            self.store.get_for_user(job.job_id, self.bob)

    def test_a_finished_job_cannot_be_finished_twice(self):
        job, _ = self.start()
        self.store.succeed(job.job_id, {"report_id": "r1"})

        with self.assertRaises(JobError):
            self.store.succeed(job.job_id, {"report_id": "r2"})

    def test_listing_is_scoped_to_session_and_user(self):
        self.start(key="a")
        self.store.succeed(self.store.list_for_session(self.session_id, self.alice)[0].job_id, {})
        self.start(key="b")

        self.assertEqual(len(self.store.list_for_session(self.session_id, self.alice)), 2)
        self.assertEqual(len(self.store.list_for_session(self.session_id, self.bob)), 0)


if __name__ == "__main__":
    unittest.main()
