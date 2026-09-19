import io
import tempfile
import unittest
import uuid
from pathlib import Path

import pandas as pd

from modules.web.app import create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory()
        self.app = create_app(self.runtime.name)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self.session_id = str(uuid.uuid4())

    def tearDown(self):
        self.runtime.cleanup()

    @staticmethod
    def _reliability_csv():
        rows = "\n".join(
            f"{index % 5 + 1},{(index + 1) % 5 + 1},{(index + 2) % 5 + 1}"
            for index in range(30)
        )
        return f"Q1,Q2,Q3\n{rows}\n".encode("utf-8")

    def _upload(self, content=None, filename="survey.csv"):
        response = self.client.post(
            "/api/upload",
            data={
                "session_id": self.session_id,
                "file": (
                    io.BytesIO(content or self._reliability_csv()),
                    filename,
                ),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_index_and_health_are_available(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("数据分析师工作台".encode("utf-8"), page.data)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")

    def test_upload_plan_execute_and_download_report(self):
        uploaded = self._upload()
        self.assertEqual(uploaded["report"]["file_type"], "csv")
        self.assertTrue(uploaded["report"]["ready_for_analysis"])

        plan = self.client.post(
            "/api/plan",
            json={
                "session_id": self.session_id,
                "file_id": uploaded["file_id"],
                "requirement": "做回归",
            },
        )
        self.assertEqual(plan.status_code, 200)
        self.assertEqual(plan.get_json()["status"], "need_clarification")
        self.assertIn("因变量", plan.get_json()["message"])

        execute = self.client.post(
            "/api/execute",
            json={
                "session_id": self.session_id,
                "file_id": uploaded["file_id"],
                "requirement": "做信度分析，题项是 Q1、Q2、Q3",
            },
        )
        self.assertEqual(execute.status_code, 200)
        payload = execute.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["results"][0]["method"], "reliability")
        self.assertTrue(payload["download_url"])

        report = self.client.get(payload["download_url"])
        self.assertEqual(report.status_code, 200)
        self.assertEqual(
            report.mimetype,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertGreater(len(report.data), 1000)

    def test_excel_upload_is_supported(self):
        buffer = io.BytesIO()
        pd.DataFrame(
            {
                "Q1": [1, 2, 3, 4],
                "Q2": [2, 3, 4, 5],
            }
        ).to_excel(buffer, index=False)
        uploaded = self._upload(buffer.getvalue(), "survey.xlsx")
        self.assertEqual(uploaded["report"]["file_type"], "xlsx")
        self.assertTrue(uploaded["report"]["ready_for_analysis"])

    def test_invalid_data_is_visible_and_blocks_execution(self):
        uploaded = self._upload(
            b"Q1,Q2\n1,2\nnot-a-number,3\n",
            "invalid.csv",
        )
        self.assertFalse(uploaded["report"]["ready_for_analysis"])
        self.assertTrue(uploaded["report"]["issues"])
        self.assertTrue(
            any(
                issue["code"] == "mixed_numeric_text"
                for issue in uploaded["report"]["issues"]
            )
        )

        execute = self.client.post(
            "/api/execute",
            json={
                "session_id": self.session_id,
                "file_id": uploaded["file_id"],
                "requirement": "做信度分析，题项是 Q1、Q2",
            },
        )
        self.assertEqual(execute.status_code, 200)
        self.assertEqual(execute.get_json()["status"], "blocked")

    def test_file_cannot_be_read_from_another_session(self):
        uploaded = self._upload()
        other_session = str(uuid.uuid4())
        response = self.client.post(
            "/api/plan",
            json={
                "session_id": other_session,
                "file_id": uploaded["file_id"],
                "requirement": "做信度分析，题项是 Q1、Q2、Q3",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("当前会话", response.get_json()["error"])

    def test_unsupported_upload_is_rejected(self):
        response = self.client.post(
            "/api/upload",
            data={
                "session_id": self.session_id,
                "file": (io.BytesIO(b"hello"), "notes.txt"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("仅支持", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
