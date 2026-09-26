import io
import tempfile
import unittest
import uuid

import pandas as pd

from modules.web.app import create_app


class WebAppTestCase(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory()
        self.app = create_app(self.runtime.name)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self.users = self.app.extensions["workbench"]["users"]
        self.token = self._new_user("analyst.zhang")
        self.session_id = self._new_session(self.token)

    def tearDown(self):
        self.runtime.cleanup()

    # -- helpers ---------------------------------------------------------

    def _new_user(self, username):
        user = self.users.create_user(username)
        return self.users.issue_token(user.user_id)

    def _auth(self, token=None):
        return {"Authorization": f"Bearer {token or self.token}"}

    def _new_session(self, token):
        response = self.client.post(
            "/api/sessions", json={"title": "测试会话"}, headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()["session_id"]

    @staticmethod
    def _reliability_csv():
        rows = "\n".join(
            f"{index % 5 + 1},{(index + 1) % 5 + 1},{(index + 2) % 5 + 1}"
            for index in range(30)
        )
        return f"Q1,Q2,Q3\n{rows}\n".encode("utf-8")

    def _upload(self, content=None, filename="survey.csv", token=None, session_id=None):
        response = self.client.post(
            "/api/upload",
            data={
                "session_id": session_id or self.session_id,
                "file": (io.BytesIO(content or self._reliability_csv()), filename),
            },
            content_type="multipart/form-data",
            headers=self._auth(token),
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def _execute(self, file_id, requirement, token=None, session_id=None):
        return self.client.post(
            "/api/execute",
            json={
                "session_id": session_id or self.session_id,
                "file_id": file_id,
                "requirement": requirement,
            },
            headers=self._auth(token),
        )


class PublicSurfaceTests(WebAppTestCase):
    def test_index_and_health_are_available(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("数据分析师工作台".encode("utf-8"), page.data)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")


class AuthenticationTests(WebAppTestCase):
    def test_every_session_scoped_endpoint_requires_a_token(self):
        cases = [
            ("post", "/api/sessions", {"json": {}}),
            ("get", "/api/sessions", {}),
            ("post", "/api/plan", {"json": {"session_id": self.session_id}}),
            ("post", "/api/execute", {"json": {"session_id": self.session_id}}),
            ("get", f"/api/files/{self.session_id}", {}),
            ("get", f"/api/jobs/{self.session_id}", {}),
            ("get", f"/api/jobs/{self.session_id}/{uuid.uuid4().hex}/result", {}),
            (
                "get",
                f"/api/reports/{self.session_id}/{uuid.uuid4().hex}/download",
                {},
            ),
        ]
        for method, path, kwargs in cases:
            with self.subTest(path=path, method=method):
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 401)

    def test_a_bad_token_is_refused(self):
        response = self.client.get(
            "/api/sessions", headers={"Authorization": "Bearer nope"}
        )
        self.assertEqual(response.status_code, 401)


class ClosedLoopTests(WebAppTestCase):
    def test_upload_plan_execute_and_download_report(self):
        uploaded = self._upload()
        self.assertEqual(uploaded["report"]["file_type"], "csv")
        self.assertTrue(uploaded["report"]["ready_for_analysis"])
        self.assertTrue(uploaded["job_id"])

        plan = self.client.post(
            "/api/plan",
            json={
                "session_id": self.session_id,
                "file_id": uploaded["file_id"],
                "requirement": "做回归",
            },
            headers=self._auth(),
        )
        self.assertEqual(plan.status_code, 200)
        self.assertEqual(plan.get_json()["status"], "need_clarification")
        self.assertIn("因变量", plan.get_json()["message"])

        execute = self._execute(uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3")
        self.assertEqual(execute.status_code, 200)
        payload = execute.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["results"][0]["method"], "reliability")
        self.assertFalse(payload["reused"])
        self.assertTrue(payload["download_url"])

        report = self.client.get(payload["download_url"], headers=self._auth())
        self.assertEqual(report.status_code, 200)
        self.assertEqual(
            report.mimetype,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertGreater(len(report.data), 1000)

    def test_results_are_traceable_to_user_session_and_job(self):
        uploaded = self._upload()
        payload = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()

        provenance = payload["results"][0]["provenance"]
        self.assertEqual(provenance["session_id"], self.session_id)
        self.assertEqual(provenance["job_id"], payload["job_id"])
        self.assertTrue(provenance["user_id"])

        jobs = self.client.get(
            f"/api/jobs/{self.session_id}", headers=self._auth()
        ).get_json()["jobs"]
        self.assertTrue(any(job["job_id"] == payload["job_id"] for job in jobs))

    def test_a_session_keeps_every_uploaded_file(self):
        first = self._upload(filename="一月.csv")
        second = self._upload(filename="二月.csv")

        listed = self.client.get(
            f"/api/files/{self.session_id}", headers=self._auth()
        ).get_json()["files"]
        self.assertEqual(
            [item["original_name"] for item in listed], ["二月.csv", "一月.csv"]
        )

        # 第一个文件依然可以继续分析，不会因为第二次上传而失联。
        execute = self._execute(first["file_id"], "做信度分析，题项是 Q1、Q2、Q3")
        self.assertEqual(execute.status_code, 200)
        self.assertEqual(execute.get_json()["status"], "completed")
        self.assertNotEqual(first["file_id"], second["file_id"])

    def test_excel_upload_is_supported(self):
        buffer = io.BytesIO()
        pd.DataFrame({"Q1": [1, 2, 3, 4], "Q2": [2, 3, 4, 5]}).to_excel(
            buffer, index=False
        )
        uploaded = self._upload(buffer.getvalue(), "survey.xlsx")
        self.assertEqual(uploaded["report"]["file_type"], "xlsx")
        self.assertTrue(uploaded["report"]["ready_for_analysis"])

    def test_invalid_data_is_visible_and_blocks_execution(self):
        uploaded = self._upload(b"Q1,Q2\n1,2\nnot-a-number,3\n", "invalid.csv")
        self.assertFalse(uploaded["report"]["ready_for_analysis"])
        self.assertTrue(
            any(
                issue["code"] == "mixed_numeric_text"
                for issue in uploaded["report"]["issues"]
            )
        )

        execute = self._execute(uploaded["file_id"], "做信度分析，题项是 Q1、Q2")
        self.assertEqual(execute.status_code, 200)
        self.assertEqual(execute.get_json()["status"], "blocked")

    def test_unsupported_upload_is_rejected(self):
        response = self.client.post(
            "/api/upload",
            data={
                "session_id": self.session_id,
                "file": (io.BytesIO(b"hello"), "notes.txt"),
            },
            content_type="multipart/form-data",
            headers=self._auth(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("仅支持", response.get_json()["error"])


class EmployeeSurfaceTests(WebAppTestCase):
    def test_employee_list_carries_each_employee_layout(self):
        listed = self.client.get("/api/employees").get_json()["employees"]

        self.assertGreaterEqual(len(listed), 2)
        by_id = {item["employee_id"]: item for item in listed}
        self.assertEqual(by_id["data_analyst"]["layout"], "analysis_split")
        self.assertTrue(by_id["data_analyst"]["available"])
        # 布局按员工不同，前端才有东西可依据。
        self.assertGreater(len({item["layout"] for item in listed}), 1)

    def test_session_response_says_which_layout_to_render(self):
        created = self.client.post(
            "/api/sessions", json={"employee": "data_analyst"}, headers=self._auth()
        ).get_json()

        self.assertEqual(created["employee"]["employee_id"], "data_analyst")
        self.assertEqual(created["employee"]["layout"], "analysis_split")

    def test_a_session_cannot_be_started_with_an_unopened_employee(self):
        for employee in ("career_coach", "does_not_exist"):
            with self.subTest(employee=employee):
                response = self.client.post(
                    "/api/sessions", json={"employee": employee}, headers=self._auth()
                )
                self.assertEqual(response.status_code, 400)


class ResultHistoryTests(WebAppTestCase):
    """之前几次分析必须还能找回来，不能只剩最新一次。"""

    def test_an_earlier_analysis_can_be_opened_again(self):
        uploaded = self._upload()
        first = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()
        self._execute(uploaded["file_id"], "对全部题项做描述统计").get_json()

        reopened = self.client.get(
            f"/api/jobs/{self.session_id}/{first['job_id']}/result",
            headers=self._auth(),
        )

        self.assertEqual(reopened.status_code, 200)
        payload = reopened.get_json()
        self.assertEqual(payload["results"], first["results"])
        self.assertEqual(payload["report_id"], first["report_id"])

    def test_the_job_listing_covers_every_analysis_in_the_session(self):
        uploaded = self._upload()
        self._execute(uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3")
        self._execute(uploaded["file_id"], "对全部题项做描述统计")

        jobs = self.client.get(
            f"/api/jobs/{self.session_id}", headers=self._auth()
        ).get_json()["jobs"]

        analyses = [job for job in jobs if job["skill"] == "data_analyst.analyze"]
        self.assertEqual(len(analyses), 2)

    def test_another_user_cannot_reopen_a_result(self):
        uploaded = self._upload()
        first = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()
        intruder = self._new_user("history.intruder")

        response = self.client.get(
            f"/api/jobs/{self.session_id}/{first['job_id']}/result",
            headers=self._auth(intruder),
        )

        self.assertEqual(response.status_code, 403)

    def test_a_job_from_another_session_is_not_reachable(self):
        uploaded = self._upload()
        first = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()
        other_session = self._new_session(self.token)

        response = self.client.get(
            f"/api/jobs/{other_session}/{first['job_id']}/result",
            headers=self._auth(),
        )

        self.assertEqual(response.status_code, 403)

    def test_an_unknown_job_is_refused(self):
        response = self.client.get(
            f"/api/jobs/{self.session_id}/{uuid.uuid4().hex}/result",
            headers=self._auth(),
        )
        self.assertEqual(response.status_code, 400)


class UsageEndpointTests(WebAppTestCase):
    def test_usage_starts_at_zero_and_is_scoped_to_the_user(self):
        mine = self.client.get("/api/usage", headers=self._auth()).get_json()
        self.assertEqual(mine["total_tokens"], 0)
        self.assertEqual(mine["by_skill"], [])

    def test_a_keyword_driven_analysis_costs_nothing(self):
        """没配模型时全走关键词表，用量必须仍是 0。"""

        uploaded = self._upload()
        self._execute(uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3")

        usage = self.client.get("/api/usage", headers=self._auth()).get_json()
        self.assertGreater(usage["jobs"], 0)
        self.assertEqual(usage["llm_calls"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_session_usage_requires_owning_the_session(self):
        intruder = self._new_user("usage.intruder")
        response = self.client.get(
            f"/api/usage/{self.session_id}", headers=self._auth(intruder)
        )
        self.assertEqual(response.status_code, 403)

    def test_usage_endpoints_require_a_token(self):
        for path in ("/api/usage", f"/api/usage/{self.session_id}"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)


class RetryTests(WebAppTestCase):
    def test_repeating_the_same_analysis_reuses_the_first_result(self):
        uploaded = self._upload()
        requirement = "做信度分析，题项是 Q1、Q2、Q3"

        first = self._execute(uploaded["file_id"], requirement).get_json()
        second = self._execute(uploaded["file_id"], requirement).get_json()

        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(second["report_id"], first["report_id"])
        self.assertEqual(second["results"], first["results"])

        reports = (
            self.app.extensions["workbench"]["root"]
            / "sessions"
            / self.session_id
            / "reports"
        )
        self.assertEqual(len(list(reports.glob("*.docx"))), 1)

    def test_a_different_requirement_produces_a_new_job(self):
        uploaded = self._upload()

        first = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()
        second = self._execute(
            uploaded["file_id"], "对全部题项做描述统计"
        ).get_json()

        self.assertNotEqual(second["job_id"], first["job_id"])
        self.assertFalse(second["reused"])


class IsolationTests(WebAppTestCase):
    """Regression cover for 产品总纲 §6: different users are fully isolated."""

    def setUp(self):
        super().setUp()
        self.intruder_token = self._new_user("intruder.li")

    def test_another_user_cannot_use_a_session_id_they_guessed(self):
        for path, payload in (
            ("/api/plan", {"file_id": uuid.uuid4().hex, "requirement": "做描述统计"}),
            ("/api/execute", {"file_id": uuid.uuid4().hex, "requirement": "做描述统计"}),
        ):
            with self.subTest(path=path):
                response = self.client.post(
                    path,
                    json={"session_id": self.session_id, **payload},
                    headers=self._auth(self.intruder_token),
                )
                self.assertEqual(response.status_code, 403)
                self.assertIn("无权访问", response.get_json()["error"])

    def test_another_user_cannot_upload_into_a_session(self):
        response = self.client.post(
            "/api/upload",
            data={
                "session_id": self.session_id,
                "file": (io.BytesIO(self._reliability_csv()), "survey.csv"),
            },
            content_type="multipart/form-data",
            headers=self._auth(self.intruder_token),
        )
        self.assertEqual(response.status_code, 403)

    def test_another_user_cannot_download_the_report(self):
        uploaded = self._upload()
        payload = self._execute(
            uploaded["file_id"], "做信度分析，题项是 Q1、Q2、Q3"
        ).get_json()

        response = self.client.get(
            payload["download_url"], headers=self._auth(self.intruder_token)
        )

        self.assertEqual(response.status_code, 403)

    def test_another_user_cannot_list_the_jobs(self):
        response = self.client.get(
            f"/api/jobs/{self.session_id}", headers=self._auth(self.intruder_token)
        )
        self.assertEqual(response.status_code, 403)

    def test_a_file_cannot_be_read_from_another_session_of_the_same_user(self):
        uploaded = self._upload()
        other_session = self._new_session(self.token)

        response = self.client.post(
            "/api/plan",
            json={
                "session_id": other_session,
                "file_id": uploaded["file_id"],
                "requirement": "做信度分析，题项是 Q1、Q2、Q3",
            },
            headers=self._auth(),
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("当前会话", response.get_json()["error"])

    def test_sessions_listing_shows_only_your_own(self):
        intruder_sessions = self.client.get(
            "/api/sessions", headers=self._auth(self.intruder_token)
        ).get_json()["sessions"]
        self.assertEqual(intruder_sessions, [])

        own = self.client.get("/api/sessions", headers=self._auth()).get_json()[
            "sessions"
        ]
        self.assertEqual([item["session_id"] for item in own], [self.session_id])


if __name__ == "__main__":
    unittest.main()
